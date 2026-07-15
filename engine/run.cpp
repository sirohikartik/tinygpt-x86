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

// Simple HTML Interface
const std::string HTML_PAGE = R"(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TinyGPT Render</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        .container { background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); width: 600px; max-width: 90%; }
        h1 { color: #333; text-align: center; margin-bottom: 1.5rem; }
        textarea { width: 100%; height: 100px; padding: 10px; border: 1px solid #ddd; border-radius: 8px; font-size: 16px; resize: none; box-sizing: border-box; outline: none; transition: border 0.3s; }
        textarea:focus { border-color: #4CAF50; }
        .controls { display: flex; justify-content: space-between; align-items: center; margin-top: 1rem; }
        .input-group { display: flex; align-items: center; gap: 8px; }
        input { width: 60px; padding: 5px; border: 1px solid #ddd; border-radius: 4px; }
        button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; transition: background 0.3s; }
        button:hover { background-color: #45a049; }
        #result { margin-top: 1.5rem; padding: 15px; background: #f9f9ff; border-radius: 8px; white-space: pre-wrap; border: 1px solid #ddd; min-height: 50px; font-size: 16px; line-height: 1.5; color: #444; }
        .loader { display: none; text-align: center; margin-top: 10px; font-style: italic; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>TinyGPT Render</h1>
        <form id="genForm">
            <textarea id="prompt" placeholder="Enter your prompt here..." required></textarea>
            <div class="controls">
                <div class="input-group">
                    <label for="max_tokens">Max Tokens:</label>
                    <input type="number" id="max_tokens" value="50">
                </div>
                <button type="submit">Generate</button>
            </div>
        </form>
        <div id="loader" class="loader">Generating response... please wait...</div>
        <div id="result">Your generated text will appear here...</div>
    </div>
    <script>
        document.getElementById('genForm').onsubmit = async (e) => {
            e.preventDefault();
            const prompt = document.getElementById('prompt').value;
            const max_tokens = document.getElementById('max_tokens').value;
            const resultDiv = document.getElementById('result');
            const loader = document.getElementById('loader');

            resultDiv.innerText = '';
            loader.style.display = 'block';

            const body = new URLSearchParams({ prompt: prompt, max_tokens: max_tokens });
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    body: body
                });
                const text = await response.text();
                resultDiv.innerText = text;
            } catch (err) {
                resultDiv.innerText = 'Error: ' + err;
            } finally {
                loader.style.display = 'none';
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

            char buffer[4096] = {0};
            int valread = read(new_socket, buffer, 4096);
            if (valread <= 0) {
                close(new_socket);
                continue;
            }

            std::string request(buffer);
            if (request.find("GET / ") != std::string::npos) {
                std::string response = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: " 
                                     + std::to_string(HTML_PAGE.length()) + "\r\n\r\n" + HTML_PAGE;
                send(new_socket, response.c_str(), response.length(), 0);
            } else if (request.find("POST /generate") != std::string::npos) {
                size_t body_pos = request.find("\r\n\r\n");
                std::string body = (body_pos == std::string::npos) ? "" : request.substr(body_pos + 4);
                
                std::string prompt = get_param(body, "prompt");
                std::string max_tokens_str = get_param(body, "max_tokens");
                int max_tokens = max_tokens_str.empty() ? 50 : std::stoi(max_tokens_str);
                
                std::cout << "Generating for prompt: " << prompt << " (max " << max_tokens << " tokens)\n";
                std::string result = model.generate(prompt, max_tokens);
                
                std::string response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: " 
                                     + std::to_string(result.length()) + "\r\n\r\n" + result;
                send(new_socket, response.c_str(), response.length(), 0);
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
