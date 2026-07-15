#include "runner.hpp"
#include "tokenizer.hpp"
#include "parser.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <omp.h>
#include <algorithm>

// Helper for URL decoding
std::string url_decode(const std::string& str) {
    std::string res;
    for (size_t i = 0; i < str.length(); ++i) {
        if (str[i] == '+') res += ' ';
        else if (str[i] == '%' && i + 2 < str.length()) {
            int value;
            std::stringstream ss;
            ss << std::hex << str.substr(i + 1, 2);
            ss >> value;
            res += static_cast<char>(value);
            i += 2;
        } else res += str[i];
    }
    return res;
}

// Helper to extract value from simple form-urlencoded body
std::string get_param(const std::string& body, const std::string& param) {
    size_t pos = body.find(param + "=");
    if (pos == std::string::npos) return "";
    size_t start = pos + param.length() + 1;
    size_t end = body.find("&", start);
    return url_decode(body.substr(start, end - start));
}

// Ultra Minimalist Black Interface
const std::string HTML_PAGE = R"(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover, interactive-widget=resizes-content">
    <title>TinyGPT</title>
    <style>
        *, *::before, *::after { box-sizing: border-box; }

        :root {
            --bg: #000;
            --fg: #fff;
            --fg-dim: #ddd;
            --border: #333;
            --btn-disabled: #333;
            --input-font: 16px;   /* never below 16px — prevents iOS/Android zoom */
            --safe-bottom: env(safe-area-inset-bottom, 0px);
        }

        html, body {
            height: 100%;
            margin: 0;
            background: var(--bg);
            color: var(--fg);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            overflow-x: hidden;
            /* Let the page scroll normally — avoids conflicts with the Android keyboard */
            overflow-y: auto;
            -webkit-text-size-adjust: 100%;
        }

        /* ── Layout ─────────────────────────────────────────────── */
        .page {
            display: flex;
            flex-direction: column;
            min-height: 100%;
            /* Bottom padding keeps content above the fixed input bar */
            padding-bottom: calc(88px + var(--safe-bottom));
        }

        .result-wrapper {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1.25rem;
        }

        #result {
            width: 100%;
            max-width: 680px;
            font-size: clamp(16px, 4.5vw, 20px);
            line-height: 1.65;
            color: var(--fg-dim);
            white-space: pre-wrap;
            text-align: center;
            word-break: break-word;
        }

        /* ── Fixed input bar ────────────────────────────────────── */
        .input-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            /* Respect Android navigation bar / iOS home indicator */
            padding: 14px 16px calc(14px + var(--safe-bottom));
            background: var(--bg);
            border-top: 1px solid #111;
            display: flex;
            align-items: center;
            gap: 12px;
            /* GPU-composited layer — smoother when keyboard slides up */
            will-change: transform;
            -webkit-backface-visibility: hidden;
        }

        input {
            flex: 1;
            min-width: 0;           /* prevents flex overflow */
            background: transparent;
            color: var(--fg);
            border: none;
            border-bottom: 1px solid var(--border);
            padding: 10px 0;
            font-size: var(--input-font);  /* 16px stops Android auto-zoom */
            outline: none;
            text-align: center;
            transition: border-color 0.25s ease;
            /* Prevent iOS grey highlight on tap */
            -webkit-tap-highlight-color: transparent;
            appearance: none;
            -webkit-appearance: none;
        }
        input::placeholder { color: #555; }
        input:focus { border-bottom-color: var(--fg); }

        button {
            flex-shrink: 0;
            width: 44px;    /* 44×44 minimum touch target (Apple/Google HIG) */
            height: 44px;
            background: var(--fg);
            color: var(--bg);
            border: none;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.18s ease, background 0.18s ease;
            -webkit-tap-highlight-color: transparent;
            /* Active state instead of hover for touch devices */
            touch-action: manipulation;
        }
        button:active { transform: scale(0.92); }
        @media (hover: hover) {
            button:hover { transform: scale(1.08); }
        }
        button:disabled {
            background: var(--btn-disabled);
            cursor: not-allowed;
            transform: none;
        }
        button svg {
            width: 20px;
            height: 20px;
            pointer-events: none;
        }

        /* ── Loader ─────────────────────────────────────────────── */
        .loader {
            position: fixed;
            top: 14px;
            right: 16px;
            font-size: 11px;
            letter-spacing: 0.06em;
            color: #555;
            display: none;
        }
    </style>
</head>
<body>
    <div class="loader" id="loader">Thinking…</div>

    <div class="page">
        <div class="result-wrapper">
            <div id="result"></div>
        </div>
    </div>

    <div class="input-bar">
        <input
            type="text"
            id="prompt"
            placeholder="Ask something…"
            autocomplete="off"
            autocorrect="on"
            spellcheck="true"
            enterkeyhint="send"
        >
        <button id="submitBtn" aria-label="Send">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                 stroke-linecap="round" stroke-linejoin="round">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 12 10 2"></polygon>
            </svg>
        </button>
    </div>

    <script>
        const promptInput = document.getElementById('prompt');
        const submitBtn   = document.getElementById('submitBtn');
        const resultDiv   = document.getElementById('result');
        const loader      = document.getElementById('loader');

        async function generate() {
            const prompt = promptInput.value.trim();
            if (!prompt) return;

            resultDiv.innerText = '';
            loader.style.display = 'block';
            submitBtn.disabled = true;
            promptInput.value = '';
            // Dismiss the Android soft keyboard after submission
            promptInput.blur();

            const body = new URLSearchParams({ prompt, max_tokens: 80 });

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    body
                });

                if (!response.ok) throw new Error('Server error ' + response.status);

                const reader  = response.body.getReader();
                const decoder = new TextDecoder();
                let fullText  = '';

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;

                    const lines = decoder.decode(value, { stream: true }).split('\n');
                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        const token = line.substring(6);
                        if (token === '[DONE]') break;
                        fullText += token;
                        resultDiv.innerText = fullText;
                        // Keep new text in view when keyboard is closed
                        resultDiv.scrollIntoView({ block: 'nearest' });
                    }
                }
            } catch (err) {
                resultDiv.innerText = 'Error: ' + err.message;
            } finally {
                loader.style.display = 'none';
                submitBtn.disabled = false;
            }
        }

        submitBtn.onclick = generate;
        // 'Enter' on the Android soft keyboard fires keydown
        promptInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); generate(); }
        });
    </script>
</body>
</html>
)";
int main() {
    omp_set_num_threads(2);
    std::cout << "OMP threads set to 2\n";

    try {
        std::map<std::string, Tensor> weights = parse_weights();
        const int num_heads = 8;
        const int blocks = 6;

        auto& emb = weights["embeddings.weight"];
        const int vocab_size = emb.shape(0);
        const int d_model = emb.shape(1);

        std::vector<std::vector<Tensor>> Qs(blocks), Ks(blocks), Vs(blocks);
        std::vector<std::vector<Tensor>> Qbs(blocks), Kbs(blocks), Vbs(blocks);
        std::vector<Tensor> gammas1(blocks), betas1(blocks);
        std::vector<Tensor> gammas2(blocks), betas2(blocks);
        std::vector<Tensor> join_weights(blocks), join_biases(blocks);
        std::vector<Tensor> ffn1_weights(blocks), ffn1_biases(blocks);
        std::vector<Tensor> ffn2_weights(blocks), ffn2_biases(blocks);

        for(int b = 0; b < blocks; b++) {
            std::string prefix = "transforms." + std::to_string(b) + ".";
            for(int h = 0; h < num_heads; h++) {
                std::string h_str = std::to_string(h);
                Qs[b].push_back(weights[prefix + "q." + h_str + ".weight"].t());
                Ks[b].push_back(weights[prefix + "k." + h_str + ".weight"].t());
                Vs[b].push_back(weights[prefix + "v." + h_str + ".weight"].t());
                Qbs[b].push_back(weights[prefix + "q." + h_str + ".bias"]);
                Kbs[b].push_back(weights[prefix + "k." + h_str + ".bias"]);
                Vbs[b].push_back(weights[prefix + "v." + h_str + ".bias"]);
            }
            gammas1[b]     = weights[prefix + "norm1.weight"];
            betas1[b]      = weights[prefix + "norm1.bias"];
            gammas2[b]     = weights[prefix + "norm2.weight"];
            betas2[b]      = weights[prefix + "norm2.bias"];
            join_weights[b] = weights[prefix + "join.weight"].t();
            join_biases[b]  = weights[prefix + "join.bias"];
            ffn1_weights[b] = weights[prefix + "ffn.0.weight"].t();
            ffn1_biases[b]  = weights[prefix + "ffn.0.bias"];
            ffn2_weights[b] = weights[prefix + "ffn.3.weight"].t();
            ffn2_biases[b]  = weights[prefix + "ffn.3.bias"];
        }

        Tensor out_weight = weights["out.weight"].t();
        Tensor out_bias   = weights["out.bias"];

        Model model(d_model, 128, vocab_size, 128, num_heads, blocks,
                    emb, out_weight, out_bias,
                    Qs, Ks, Vs,
                    Qbs, Kbs, Vbs,
                    join_weights, join_biases,
                    gammas1, betas1,
                    gammas2, betas2,
                    ffn1_weights, ffn1_biases,
                    ffn2_weights, ffn2_biases,
                    "gpt2_vocab.json", "merges.txt");

        std::cout << "Model initialized successfully!\n";

        int server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd == 0) {
            perror("Socket failed");
            return 1;
        }

        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in address;
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        
        const char* port_env = std::getenv("PORT");
        int port = port_env ? std::stoi(port_env) : 8080;
        address.sin_port = htons(port);

        if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
            perror("Bind failed");
            return 1;
        }

        if (listen(server_fd, 3) < 0) {
            perror("Listen failed");
            return 1;
        }

        std::cout << "Server running on http://0.0.0.0:" << port << "\n";

        while (true) {
            int addrlen = sizeof(address);
            int new_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen);
            if (new_socket < 0) continue;

            std::string request_data;
            char buffer[1024];
            while (true) {
                int valread = read(new_socket, buffer, 1024);
                if (valread <= 0) break;
                request_data.append(buffer, valread);
                if (request_data.find("\r\n\r\n") != std::string::npos) {
                    size_t cl_pos = request_data.find("Content-Length: ");
                    if (cl_pos != std::string::npos) {
                        size_t end_line = request_data.find("\r\n", cl_pos);
                        int content_length = std::stoi(request_data.substr(cl_pos + 16, end_line - (cl_pos + 16)));
                        size_t body_start = request_data.find("\r\n\r\n") + 4;
                        if (request_data.length() >= body_start + content_length) {
                            break; 
                        }
                    } else {
                        if (request_data.find("GET ") != std::string::npos) break;
                    }
                }
            }

            if (request_data.empty()) {
                close(new_socket);
                continue;
            }

            if (request_data.find("GET / ") != std::string::npos) {
                std::string response = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: " 
                                     + std::to_string(HTML_PAGE.length()) + "\r\n\r\n" + HTML_PAGE;
                send(new_socket, response.c_str(), response.length(), 0);
            } else if (request_data.find("POST /generate") != std::string::npos) {
                size_t body_pos = request_data.find("\r\n\r\n");
                std::string body = (body_pos == std::string::npos) ? "" : request_data.substr(body_pos + 4);
                
                std::string prompt = get_param(body, "prompt");
                std::string max_tokens_str = get_param(body, "max_tokens");
                int max_tokens = max_tokens_str.empty() ? 80 : std::stoi(max_tokens_str);
                
                if (prompt.empty()) {
                    std::cout << "[Server] Error: Prompt was empty in request body!\n";
                    std::string response = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n";
                    send(new_socket, response.c_str(), response.length(), 0);
                    close(new_socket);
                    continue;
                }

                std::cout << "[Server] Request received. Prompt: " << prompt << "\n";
                
                std::string header = "HTTP/1.1 200 OK\r\n"
                                    "Content-Type: text/event-stream\r\n"
                                    "Cache-Control: no-cache\r\n"
                                    "Connection: keep-alive\r\n"
                                    "Access-Control-Allow-Origin: *\r\n\r\n";
                send(new_socket, header.c_str(), header.length(), 0);

                model.generate_stream(prompt, max_tokens, [&](std::string token) {
                    std::string sse_chunk = "data: " + token + "\n\n";
                    send(new_socket, sse_chunk.c_str(), sse_chunk.length(), 0);
                });

                std::string end_chunk = "data: [DONE]\n\n";
                send(new_socket, end_chunk.c_str(), end_chunk.length(), 0);
            } else {
                std::string response = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
                send(new_socket, response.c_str(), response.length(), 0);
            }
            close(new_socket);
        }

    } catch(const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
