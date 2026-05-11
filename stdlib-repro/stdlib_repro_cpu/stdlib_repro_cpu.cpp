#include <torch/all.h>

#include <iomanip>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

uint64_t fnv1a(std::string const& text, uint64_t seed) {
  uint64_t hash = seed;
  for (unsigned char c : text) {
    hash ^= c;
    hash *= 1099511628211ull;
  }
  return hash;
}

std::vector<std::string> regex_includes(std::string const& code) {
  static const std::regex include_regex(R"(^\s*#include\s+[<"]([^>"]+)[>"])");

  std::vector<std::string> includes;
  size_t line_begin = 0;
  while (line_begin <= code.size()) {
    auto line_end = code.find('\n', line_begin);
    if (line_end == std::string::npos) {
      line_end = code.size();
    }

    const auto line = code.substr(line_begin, line_end - line_begin);
    std::smatch match;
    if (std::regex_search(line, match, include_regex)) {
      includes.push_back(match[1].str());
    }

    if (line_end == code.size()) {
      break;
    }
    line_begin = line_end + 1;
  }
  return includes;
}

std::string stream_hex_digest(std::string const& text) {
  const auto state_0 = fnv1a(text, 0xc6a4a7935bd1e995ull);
  const auto state_1 = fnv1a(text, 0x9e3779b97f4a7c15ull);

  std::ostringstream stream;
  stream << std::hex << std::setfill('0') << std::setw(16) << state_0 << std::setw(16) << state_1;
  return stream.str();
}

std::string exercise_stdlib(std::string const& code, int64_t iters) {
  TORCH_CHECK(iters >= 0, "iters must be non-negative");

  std::string digest;
  for (int64_t i = 0; i < iters; ++i) {
    const auto includes = regex_includes(code);

    std::string joined;
    for (auto const& include : includes) {
      joined += include;
      joined += "$";
    }
    joined += "#";
    joined += code;
    joined += "#";
    joined += std::to_string(i);

    digest = stream_hex_digest(joined);
  }
  return digest;
}

}  // namespace

std::string relu_and_probe(
    torch::Tensor& out,
    torch::Tensor const& input,
    std::string const& code,
    int64_t iters) {
  TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");
  TORCH_CHECK(out.device().is_cpu(), "out must be a CPU tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(out.dtype() == torch::kFloat32, "out must be float32");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(out.numel() == input.numel(), "input and out must have the same number of elements");

  auto const* input_ptr = input.data_ptr<float>();
  auto* out_ptr = out.data_ptr<float>();
  for (int64_t i = 0; i < input.numel(); ++i) {
    out_ptr[i] = input_ptr[i] > 0.0f ? input_ptr[i] : 0.0f;
  }

  return exercise_stdlib(code, iters);
}
