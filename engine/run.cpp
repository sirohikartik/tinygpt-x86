#include "runner.hpp"
#include "tokenizer.hpp"
#include "parser.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
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

// Beautiful Dark Theme Interface
const std::string HTML_PAGE = R"(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TinyGPT Render | Dark Edition</title>
    <style>
        :root {
            --bg-color: #0a0a0a;
            --container-bg: #121212;
            --accent-color: #4CAF50;
            --text-primary: #e0e0e0;
            --text-secondary: #a0a0a0;
            --input-bg: #1e1e1e;
            --border-color: #333;
        }
        body { 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            background-color: var(--bg-color); 
            color: var(--text-primary);
            margin: 0; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            min-height: 100vh; 
        }
        .container { 
            background: var(--container-bg); 
            padding: 2.5rem; 
            border-radius: 20px; 
            box-shadow: 0 10px 50px rgba(0,0,0,0.5); 
            width: 700px; 
            max-width: 90%; 
            border: 1px solid var(--border-color);
        }
        h1 { 
            color: var(--text-primary); 
            text-align: center; 
            margin-bottom: 2rem; 
            font-weight: 300; 
            letter-spacing: -1px; 
        }
        h1 span { color: var(--accent-color); font-weight: 700; }
        textarea { 
            width: 100%; 
            height: 120px; 
            padding: 15px; 
            background: var(--input-bg);
            color: var(--text-primary);
            border: 1px solid var(--border-color); 
            border-radius: 12px; 
            font-size: 16px; 
            resize: none; 
            box-sizing: border-box; 
            outline: none; 
            transition: all 0.3s ease; 
        }
        textarea:focus { border-color: var(--accent-color); box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2); }
        .controls { display: flex; justify-content: space-between; align-items: center; margin-top: 1.5rem; }
        .input-group { display: flex; align-items: center; gap: 10px; color: var(--text-secondary); font-size: 14px; }
        input { 
            width: 60px; 
            padding: 6px; 
            background: var(--input-bg);
            color: var(--text-primary);
            border: 1px solid var(--border-color); 
            border-radius: 6px; 
            text-align: center;
        }
        button { 
            background-color: var(--accent-color); 
            color: white; 
            padding: 12px 24px; 
            border: none; 
            border-radius: 10px; 
            cursor: pointer; 
            font-size: 16px; 
            font-weight: 600;
            transition: all 0.3s ease; 
        }
        button:hover { background-color: #45a049; transform: translateY(-2px); }
        button:disabled { background-color: #333; cursor: not-allowed; transform: none; }
        #result { 
            margin-top: 2rem; 
            padding: 20px; 
            background: var(--input-bg); 
            border-radius: 12px; 
            white-space: pre-wrap; 
            border: 1px solid var(--border-color); 
            min-height: 100px; 
            font-size: 16px; 
            line-height: 1.6; 
            color: var(--text-primary); 
            transition: all 0.3s ease;
        }
        .loader { 
            display: none; 
            text-align: center; 
            margin-top: 15px; 
            font-size: 14px;
            color: var(--text-secondary); 
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { opacity: 0.5; }
            50% { opacity: 1; }
            100% { opacity: 0.5; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>TinyGPT <span>Render</span></h1>
        <form id="genForm">
            <textarea id="prompt" placeholder="What is the secret of the universe?..." required></textarea>
            <div class="controls">
                <div class="input-group">
                    <label for="max_tokens">Max Tokens:</label>
                    <input type="number" id="max_tokens" value="50">
                </div>
                <button type="submit" id="submitBtn">Generate</button>
            </div>
        </form>
        <div id="loader" class="loader">Thinking...</div>
        <div id="result">Response will appear here...</div>
    </div>
    <script>
        document.getElementById('genForm').onsubmit = async (e) => {
            e.preventDefault();
            const prompt = document.getElementById('prompt').value;
            const max_tokens = document.getElementById('max_tokens').value;
            const resultDiv = document.getElementById('result');
            const loader = document.getElementById('loader');
            const submitBtn = document.getElementById('submitBtn');

            resultDiv.innerText = '';
            loader.style.display = 'block';
            submitBtn.disabled = true;

            const body = new URLSearchParams({ prompt: prompt, max_tokens: max_tokens });
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    body: body
                });

                if (!response.ok) throw new Error('Server Error');

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullText = '';

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value, { stream: true });
                    // The server sends "data: token\n\n"
                    const lines = chunk.split('\\n');
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const token = line.substring(6);
                            fullText += token;
                            resultDiv.innerText = fullText;
                        }
                    }
                }
            } catch (err) {
                resultDiv.innerText = 'Error: ' + err.message;
            } finally {
                loader.style.display = 'none';
                submitBtn.disabled = false;
            }
        };
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
        address.sin_port = htons(8080);

        if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
            perror("Bind failed");
            return 1;
        }

        if (listen(server_fd, 3) < 0) {
            perror("Listen failed");
            return 1;
        }

        std::cout << "Server running on http://localhost:8080\n";

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
                    // Check if we have the full body based on Content-Length
                    size_t cl_pos = request_data.find("Content-Length: ");
                    if (cl_pos != std::string::npos) {
                        size_t end_line = request_data.find("\r\n", cl_pos);
                        int content_length = std::stoi(request_data.substr(cl_pos + 16, end_line - (cl_pos + 16)));
                        size_t body_start = request_data.find("\r\n\r\n") + 4;
                        if (request_data.length() >= body_start + content_length) {
                            break; 
                        }
                    } else {
                        // No content length, probably a GET request
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
                int max_tokens = max_tokens_str.empty() ? 50 : std::stoi(max_tokens_str);
                
                if (prompt.empty()) {
                    std::cout << "[Server] Error: Prompt was empty in request body!\n";
                    std::string response = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n";
                    send(new_socket, response.c_str(), response.length(), 0);
                    close(new_socket);
                    continue;
                }

                std::cout << "[Server] Received prompt: " << prompt << " (Max tokens: " << max_tokens << ")\n";
                
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
