---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}---

This is a tiny Kernel Hub repro for C++ standard-library runtime crashes in packaged kernels.

It exposes a CPU op that runs:

- `std::regex` include parsing
- `std::ostringstream`/`std::hex` hash formatting
- a simple ReLU copy so the op has a normal tensor path

## How to use
{% if functions %}

```python
from kernels import get_kernel

kernel_module = get_kernel("{{ repo_id }}", backend="cpu", trust_remote_code=True)
out, digest = kernel_module.relu_and_probe(x, iters=1000)
```
{% else %}

Usage example not available.
{% endif %}

## Available functions
{% if functions %}
{% for func in functions %}
- `{{ func }}`
{% endfor %}
{% else %}

Function list not available.
{% endif %}
