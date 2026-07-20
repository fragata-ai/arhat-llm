# Arhat LLM

Arhat LLM is a library of deep neural network primitives designed for implementation of LLM inference.
Arhat LLM can be used either in combination with [llama.cpp](https://github.com/ggml-org/llama.cpp) 
framework or as a standalone library.
Integration with llama.cpp is facilitated via a specialized Arhat backend for GGML.

Arhat LLM targets Intel Arc GPU hardware and is built on top of 
[oneAPI Deep Neural Network Library](https://github.com/uxlfoundation/oneDNN) (oneDNN).

In this distribution, the Arhat LLM code is merged with the code of llama.cpp (Release b9999).

The README page of Arhat LLM is located [here](./README-arhat.md).

The README page of llama.cpp distribution is located [here](./README-llama-cpp.md).

The code of Arhat library is located in the subdirectory `arhat`.
The code of the Arhat backend for GGML is located in the subdirectory `ggml\src\arhat-ggml`.


## Licensing

Use of Arhat LLM is governed by MIT License. Refer to the [License](arhat/LICENSE)
file for the full license text and copyright notices.

Use of the llama.cpp framework included in this distribution is governed by the MIT Licence.
Refer to the [License](./LICENSE) file for the full license text and copyright notice.

This distribution includes third party software located in the `vendor` subdirectory and
governed by separate license terms attached to the respective parts of the source code.


## Contributing

We welcome your feedback in the form of questions, bug reports, and feature requests.
To submit the feedback, please use the Github issues page.

We do not accept pull requests. This public source is exported from our internal development
system which does not support the workflow for external pull requests.




