#include <vector>
#include <iostream>

template <typename T, typename... Args>
class Tensor {
public:
  // Nested type representing the shape of the tensor
  using shape_type = std::vector<size_t, sizeof...(Args) + 1>;

  // Constructor that deduces the shape from arguments
  Tensor(T value, Args&&... args) : data_(value), dims_({value, args...}) {}

  // Access element at specific indices (assuming row-major order)
  T& operator()(const shape_type& indices) {
    if (indices.size() != dims_.size()) {
      throw std::invalid_argument("Indices size must match tensor rank");
    }

    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      if (indices[i] >= dims_[i]) {
        throw std::out_of_range("Index out of bounds");
      }
      offset += indices[i] * std::accumulate(dims_.begin() + i + 1, dims_.end(), 1, std::multiplies<size_t>());
    }
    return data_[offset];
  }

private:
  // Stores the underlying data element
  T data_;
  // Stores the shape of the tensor
  shape_type dims_;

  // Reference operator for initializer list compatibility
  shape_type& operator&() {
    return dims_;
  }
};

int main() {
  try {
    // Example 1: Using separate initialization for data type and dimensions
    Tensor<int, 2, 5> t1(2, 3, 5, 2, 1);

    // Example 2: Using multidimensional initializer list (C++17 and above)
    Tensor<int, 2, 5> t2 = { {2, 3, 5, 2, 1}, {2, 3, 5, 2, 1} };

    // Accessing elements
    int element1 = t1({1, 2}); // Access element at (1, 2)
    int element2 = t2({0, 3}); // Access element at (0, 3)

    // Print the elements (assuming you have an appropriate print function)
    // print(element1);
    // print(element2);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }

  return 0;
}