#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <stdlib.h> // rand

#include "memory.hpp"


template<typename T>
class TensorData {

    using Self = TensorData<T>;

public:

    class TensorRef
    {
    public:
        TensorRef(SmartPointer<T> t = nullptr, size_t index = -1) : td {t}, index {index} {
            
        }

        TensorRef& operator=(T v) {
            if (!isNull())
                *(td.ptr() + index) = v;
            return *this;
        }

        bool isNull() const {
            return !td.isValid();
        }

        T item() {
            if (isNull())
                return T{};

            return *(td.ptr() + index);
        }

        inline auto const& ptr() const {
            return td;
        }

    private:
        SmartPointer<T> td;
        size_t index;
    };


public:

    TensorData(size_t count) : TensorData(count, {}) {}

    TensorData(size_t count, T def) : 
        count {count}, 
        buf {
            SmartPointer<T>::fromRaw(mem::alloc<T>(count, def)) 
        } {
        
        }

    void* bufptr(size_t indx) {
        if (!indexInRange(indx))
            return nullptr;
        else return buf.ptr() + indx;
    }

    TensorRef operator[] (size_t indx) {
        SmartPointer<T> bff = nullptr;
        if (indexInRange(indx))
            bff = buf;

        return TensorRef(bff, indx);
    }

    bool indexInRange(size_t indx) {
         if (indx < 0 || indx >= count)
            return false;
        return true;
    }

    static Self fromShape() {
        return {};
    }

    Self operator+(Self& t) {
        return {};
    }

    Self operator*(Self& t) {
        return {};
    }

    size_t elements() const {
        return count;
    }

    SmartPointer<T> const& bufferRef() const {
        return buf;
    }

    SmartPointer<T> buffer() {
        return buf;
    }

private:

    SmartPointer<T> buf = nullptr;
    size_t count;
};



template <typename T>
class TensorBase {

    using Self = TensorBase<T>;
    using ShapeType = std::vector<size_t>;

public:

    static TensorBase ones(ShapeType s) {
        return TensorBase(s).initWith(1);
    }

    static TensorBase zeros(ShapeType s) {
        return TensorBase(s).initWith(0);
    }

    static TensorBase<T> randn(ShapeType s) {
        auto t = TensorBase(s);

        auto ptr   = t.data.get().buffer();
        auto _ptr  = ptr.ptr();

        auto& data = t;
        
        for (auto i = 0; i < data.elements(); ++i)
            _ptr[i] = static_cast<float>(::rand()) / static_cast<float>(RAND_MAX);

        return t;
    }

    TensorBase(ShapeType s) : 
        shape {s},
        data { shapeEntryCount(shape) },

        children {} {
            
        }

    Self& initWith(T t) {
        auto ptr = data.get().buffer();
        auto _ptr = ptr.ptr();
        
        for (auto i = 0; i < data.get().elements(); ++i)
            _ptr[i] = t;

        return *this;
    }

    void set(ShapeType indx, T value) {
        get(indx) = value;
    }

    typename TensorData<T>::TensorRef get(ShapeType indx) {
        return data.get()[computeIndex(indx)];
    }

    auto shapeEntryCount(ShapeType s) {
        auto r = s[0];
        for (int i = 1; i < s.size(); ++i)
            r *= s[i];

        return r;
    }

    ShapeType getShape() const {
        return shape;
    }

    /*
    
    example of this algorithm
    const v = [2, 1, 4];
    const s = [3, 2, 5];

    v0*s1*s2 + v1*s2 + v2

    */
    size_t computeIndex(ShapeType indx) {
        // TODO: set caching system
        //
        if (indx.size() != shape.size())
            return -1;
        
        for (size_t i = 0; i < shape.size(); ++i) {
            if (indx[i] >= shape[i] || indx[i] < 0)
                return -1;
        }

        size_t index = 0;

        for (size_t i = 0; i < shape.size() - 1; ++i) {
            auto currPos = shape.begin() + 1 + i;
            size_t product = 1;
            for (auto i = currPos; currPos != shape.end(); currPos++)
                product *= *currPos;

            index += product * indx[i];
        }
        

        index += indx[indx.size() - 1];
        return index;
    }

    Self operator+(Self const& r) {
        return Self{this->value() + r.value(), {*this, r}};
    }

    Self operator*(Self const& r) {
        return Self{this->value() * r.value(), {*this, r}};
    }

    inline TensorData<T> value() const {
        return data.value();
    }

    const auto& dataPtr() const {
        return data;
    }

    const auto& gradPtr() const {
        return grad;
    }

    T var() const {
        auto m = mean();
        auto acc = T{0};
        
        forEach([&](T x) {
            acc += ::pow(x - m, 2);
        });

        return acc / elements();
    }

    T std() const {
        return ::sqrt(var());
    }

    T mean() const {
        auto acc = T{0};

        forEach([&acc](T el) {
            acc += el;
        });

        return acc / elements();
    }

    
    template <typename ConsumerFunction>
    void forEach(ConsumerFunction fn) const {
        auto d      = data.get();
        auto bufptr = d.buffer().ptr();
        
        for (size_t i = 0; i < d.elements(); ++i)
            fn(bufptr[i]);
    }

    
    auto elements() const {
        return data.get().elements();
    }

private:
    ShapeType shape;

    SmartPointer<TensorData<T>> data {0};
    SmartPointer<TensorData<T>> grad {0};

    std::vector<Self> children;
};

using Tensor = TensorBase<float>;


int main() {

    // Tensor t1 = Tensor::zeros({100, 100});
    Tensor t2 = Tensor::randn({100, 100});

    std::cout << "MEAN: " << t2.mean() << std::endl;
    std::cout << "VAR: " << t2.var() << std::endl;
    

    
    // component wise.
    // Tensor t3 = t1 + t2;


    //std::cout << t1.dataPtr().referenceCount() << std::endl;


   



    return 0;
}