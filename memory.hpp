#pragma once

struct mem {
    template<typename T>
    static T* alloc(size_t size, T def) {
        auto b = new T[size];
        for (size_t i = 0; i < size; ++i)
            b[i] = def;

        return b;
    }
};


template<typename T>
class SmartPointer {

public:

    static SmartPointer<T> fromRaw(T* ptr) {
        return ptr;
    }

    SmartPointer(T* ptr) : data {ptr}, refCount{createRefCount()} {acquire();}

    SmartPointer() : SmartPointer(nullptr) {}

    SmartPointer(T d) : data {new T{d}}, refCount{createRefCount()} {
        acquire();
    }

    SmartPointer(const SmartPointer& ptr) : data {ptr.data}, refCount {ptr.refCount} {
        acquire();
    }

    int* createRefCount() {
        if (!data)
            return nullptr;

        return new int{0};;
    }

    ~SmartPointer() {
        this->release();
    }

    SmartPointer& operator=(const SmartPointer<T>& p) {

        this->release();

        data     = p.data;        
        refCount = p.refCount;

        this->acquire();

        return *this;
    }

    void acquire() {
        if (refCount) (*refCount)++;
    }

    void release() {
        if (!refCount)
            return;

        --(*refCount);
        if (*refCount <= 0)
            freeResource();
    }

    T value() {
        return *data;
    }

    T value() const {
        return *data;
    }

    T* ptr() {
        return data;
    }

    T& get() {
        return *data;
    }

    bool isValid() const {
        return data != nullptr;
    }

    void set(T val) {
        if (!data)
            data = new T{val};
        else
            *data = val;
    }

    inline int referenceCount() const {

        return *refCount;
    }

private:
    void freeResource() {
       
        delete refCount;
        delete data;


        refCount = nullptr;
        data     = nullptr;
    }

private:
    T* data       = nullptr;
    int* refCount = nullptr;
};
