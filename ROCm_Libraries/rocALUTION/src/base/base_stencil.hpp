/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_BASE_STENCIL_HPP_
#define ROCALUTION_BASE_STENCIL_HPP_

#include "base_rocalution.hpp"

namespace rocalution {

template <typename ValueType>
class BaseVector;
template <typename ValueType>
class HostVector;
template <typename ValueType>
class HIPAcceleratorVector;

template <typename ValueType>
class HostStencilLaplace2D;
template <typename ValueType>
class HIPAcceleratorStencil;
template <typename ValueType>
class HIPAcceleratorStencilLaplace2D;

/// Base class for all host/accelerator stencils
template <typename ValueType>
class BaseStencil
{
    public:
    BaseStencil();
    virtual ~BaseStencil();

    /// Return the number of rows in the stencil
    int GetM(void) const;
    /// Return the number of columns in the stencil
    int GetN(void) const;
    /// Return the dimension of the stencil
    int GetNDim(void) const;
    /// Return the nnz per row
    virtual int GetNnz(void) const = 0;

    /// Shows simple info about the object
    virtual void Info(void) const = 0;
    /// Return the stencil format id (see stencil_formats.hpp)
    virtual unsigned int GetStencilId(void) const = 0;
    /// Copy the backend descriptor information
    virtual void set_backend(const Rocalution_Backend_Descriptor local_backend);
    // Set the grid size
    virtual void SetGrid(int size);

    /// Apply the stencil to vector, out = this*in;
    virtual void Apply(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const = 0;
    /// Apply and add the stencil to vector, out = out + scalar*this*in;
    virtual void ApplyAdd(const BaseVector<ValueType>& in,
                          ValueType scalar,
                          BaseVector<ValueType>* out) const = 0;

    protected:
    /// Number of rows
    int ndim_;
    /// Number of columns
    int size_;

    /// Backend descriptor (local copy)
    Rocalution_Backend_Descriptor local_backend_;

    friend class BaseVector<ValueType>;
    friend class HostVector<ValueType>;
    friend class AcceleratorVector<ValueType>;
    friend class HIPAcceleratorVector<ValueType>;
};

template <typename ValueType>
class HostStencil : public BaseStencil<ValueType>
{
    public:
    HostStencil();
    virtual ~HostStencil();
};

template <typename ValueType>
class AcceleratorStencil : public BaseStencil<ValueType>
{
    public:
    AcceleratorStencil();
    virtual ~AcceleratorStencil();

    /// Copy (accelerator stencil) from host stencil
    virtual void CopyFromHost(const HostStencil<ValueType>& src) = 0;

    /// Copy (accelerator stencil) to host stencil
    virtual void CopyToHost(HostStencil<ValueType>* dst) const = 0;
};

template <typename ValueType>
class HIPAcceleratorStencil : public AcceleratorStencil<ValueType>
{
    public:
    HIPAcceleratorStencil();
    virtual ~HIPAcceleratorStencil();
};

} // namespace rocalution

#endif // ROCALUTION_BASE_STENCIL_HPP_
