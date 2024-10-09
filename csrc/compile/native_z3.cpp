// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "native_z3.h"

#define USE_C10D_NCCL

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/cuda/nccl.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <torch/csrc/distributed/c10d/SymmetricMemory.hpp>

namespace n3z {

static c10::intrusive_ptr<c10d::ProcessGroup> process_group = nullptr;

class DSParam {
public:
    DSParam(long id,
            std::vector<int64_t> ds_shape,
            at::Tensor ds_tensor,
            at::Tensor grad_buffer,
            bool persistent)
        : id_(id),
          shape_(std::move(ds_shape)),
          ds_tensor_(ds_tensor),
          grad_buffer_(grad_buffer),
          persistent_(persistent)
    {
    }

    long getId() const { return id_; }
    std::vector<int64_t> getShape() const { return shape_; }
    at::Tensor getDSTensor() const { return ds_tensor_; }
    at::Tensor getGradBuffer() const { return grad_buffer_; }
    bool isPersistent() const { return persistent_; }

private:
    long id_;
    std::vector<int64_t> shape_;
    at::Tensor ds_tensor_;
    at::Tensor grad_buffer_;
    bool persistent_;
};

class GraphOpStates {
public:
    GraphOpStates() {}
    ~GraphOpStates() {}

    void registerOpNArgs(const std::string& op_name, long n_args)
    {
        op_n_args_[op_name] = n_args;
        args_counter_[op_name] = n_args;
    }

    void resetArgCounter()
    {
        // std::cout << "resetArgCounter size op_n_args_ " << op_n_args_.size() << std::endl;

        for (const auto& it : op_n_args_) {
            assert(hasKey(op_n_args_, it.first));
            args_counter_[it.first] = op_n_args_.at(it.first);
        }
    }

    void decrementArgCounter(const std::string& op_name)
    {
        // std::cout << "decrementArgCounter " << op_name << std::endl;

        assert(hasKey(args_counter_, op_name));
        if (args_counter_.at(op_name) == 0) return;
        args_counter_[op_name]--;
    }

    long getArgCounter(const std::string& op_name) const
    {
        assert(hasKey(args_counter_, op_name));
        return args_counter_.at(op_name);
    }

    bool isArgCounterZero(const std::string& op_name) const
    {
        assert(hasKey(args_counter_, op_name));
        return args_counter_.at(op_name) == 0;
    }

private:
    std::unordered_map<std::string, size_t> op_n_args_;
    std::unordered_map<std::string, size_t> args_counter_;
};

c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> getSymmMemWorkspace(int64_t size)
{
    c10::Device device = c10::Device(c10::kCUDA, c10::cuda::current_device());
    std::vector<int64_t> sizes = {size};
    std::vector<int64_t> strides = {1};
    at::Tensor sym_mem_ws = c10d::symmetric_memory::empty_strided_p2p(
        {size}, {1}, c10::ScalarType::Byte, device, process_group->getGroupName(), std::nullopt);
    return c10d::symmetric_memory::rendezvous(sym_mem_ws);
}

class DSParamRegistry {
public:
    DSParamRegistry() {}
    ~DSParamRegistry() {}

    void registerParam(long ds_id,
                       const std::vector<int64_t>& ds_shape,
                       at::Tensor ds_tensor,
                       at::Tensor grad_buffer,
                       bool persistent)
    {
        params_.emplace(ds_id, DSParam(ds_id, ds_shape, ds_tensor, grad_buffer, persistent));
    }

    void registerGatheredParam(long ds_id, at::Tensor ds_tensor)
    {
        gathered_params_.emplace(ds_id, ds_tensor);
    }

    void unregisterGatheredParam(long ds_id) { gathered_params_.erase(ds_id); }

    const std::unordered_map<long, DSParam>& getParams() const { return params_; }

    const DSParam& getParam(long ds_id) const { return params_.at(ds_id); }
    const size_t getNumParams() const { return params_.size(); }
    const at::Tensor& getGatheredParam(long ds_id) const { return gathered_params_.at(ds_id); }
    bool hasGatheredParam(long ds_id) const { return hasKey(gathered_params_, ds_id); }
    // bool useSymmMem() const { return use_symm_mem_; }

private:
    std::unordered_map<long, DSParam> params_;
    std::unordered_map<long, at::Tensor> gathered_params_;
};

class ReduceTask {
public:
    ReduceTask(long ds_id, at::Tensor send_buf) : ds_id_(ds_id), send_buf_(std::move(send_buf)) {}

    long getDSId() const { return ds_id_; }
    at::Tensor getSendBuf() const { return send_buf_; }

private:
    long ds_id_;
    at::Tensor send_buf_;
};

class ReduceBucket {
public:
    ReduceBucket(int64_t size, at::ScalarType scalar_type) : size_(size), scalar_type_(scalar_type)
    {
        buffer_ = torch::empty({size}, at::TensorOptions().dtype(scalar_type).device(at::kCUDA));
        offset_ = 0;
    }

    int64_t getSize() const { return size_; }
    int64_t getOffset() const { return offset_; }
    at::Tensor getBuffer() const { return buffer_; }
    at::ScalarType getScalarType() const { return scalar_type_; }

    void reserve(int64_t size)
    {
        if (size > size_) {
            buffer_ =
                torch::empty({size}, at::TensorOptions().dtype(scalar_type_).device(at::kCUDA));
            size_ = size;
        }
    }

    at::Tensor allocate(int64_t numel)
    {
        if (offset_ + numel > size_) {
            throw std::runtime_error("Buffer size exceeds the reduce bucket size");
        }

        at::Tensor result = buffer_.index({torch::indexing::Slice(offset_, offset_ + numel)});
        offset_ += numel;
        return result;
    }

    bool shouldFlush(int64_t numel) { return offset_ > 0 && offset_ + numel > size_; }

    void reset() { offset_ = 0; }

private:
    int64_t size_;
    int64_t offset_;
    at::Tensor buffer_;
    at::ScalarType scalar_type_;
};

class DoubleBufferedReduceBucket {
public:
    DoubleBufferedReduceBucket(int64_t initial_bucket_size, bool enable_double_buffer)
        : initial_bucket_size_(initial_bucket_size), enable_double_buffer_(enable_double_buffer)
    {
    }

    void swap(at::ScalarType scalar_type, at::cuda::CUDAStream stream)
    {
        assert(hasKey(current_buffer_, scalar_type));
        assert(hasKey(current_buffer_events_, scalar_type));

        current_buffer_.at(scalar_type)->reset();
        current_buffer_events_.at(scalar_type)->record(stream);

        if (enable_double_buffer_) {
            assert(hasKey(shadow_buffer_, scalar_type));
            assert(hasKey(shadow_buffer_events_, scalar_type));

            auto tmp = current_buffer_.at(scalar_type);
            current_buffer_[scalar_type] = shadow_buffer_.at(scalar_type);
            shadow_buffer_[scalar_type] = tmp;

            auto tmp_event = current_buffer_events_.at(scalar_type);
            current_buffer_events_[scalar_type] = shadow_buffer_events_.at(scalar_type);
            shadow_buffer_events_[scalar_type] = tmp_event;
        }
    }

    std::shared_ptr<ReduceBucket> getBuffer(at::ScalarType scalar_type)
    {
        if (!hasKey(current_buffer_, scalar_type)) {
            current_buffer_[scalar_type] =
                std::make_shared<ReduceBucket>(initial_bucket_size_, scalar_type);
            current_buffer_events_[scalar_type] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);

            if (enable_double_buffer_) {
                shadow_buffer_[scalar_type] =
                    std::make_shared<ReduceBucket>(initial_bucket_size_, scalar_type);
                shadow_buffer_events_[scalar_type] =
                    std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
            }
        }

        return current_buffer_.at(scalar_type);
    }

    std::shared_ptr<at::cuda::CUDAEvent> getEvent(at::ScalarType scalar_type)
    {
        assert(hasKey(current_buffer_events_, scalar_type));
        return current_buffer_events_.at(scalar_type);
    }

private:
    int64_t initial_bucket_size_;
    bool enable_double_buffer_;
    std::unordered_map<at::ScalarType, std::shared_ptr<ReduceBucket>> current_buffer_;
    std::unordered_map<at::ScalarType, std::shared_ptr<ReduceBucket>> shadow_buffer_;
    std::unordered_map<at::ScalarType, std::shared_ptr<at::cuda::CUDAEvent>> current_buffer_events_;
    std::unordered_map<at::ScalarType, std::shared_ptr<at::cuda::CUDAEvent>> shadow_buffer_events_;
};

ncclDataType_t get_nccl_data_type(at::ScalarType scalar_type)
{
    switch (scalar_type) {
        case at::kFloat: return ncclFloat;
        case at::kHalf: return ncclHalf;
        case at::kDouble: return ncclDouble;
        case at::kBFloat16: return ncclBfloat16;
        case at::kLong: return ncclInt64;
        case at::kInt: return ncclInt;
        case at::kChar: return ncclInt8;
        default: throw std::runtime_error("Unsupported scalar type");
    }
}

class CustomOpExecutor {
public:
    CustomOpExecutor(c10::intrusive_ptr<c10d::ProcessGroup> process_group,
                     std::shared_ptr<DSParamRegistry> param_registry,
                     std::shared_ptr<DoubleBufferedReduceBucket> reduce_buckets,
                     std::vector<long> ds_ids,
                     ncclComm_t nccl_comm,
                     at::cuda::CUDAStream comm_stream)
        : process_group_(process_group),
          param_registry_(std::move(param_registry)),
          reduce_buckets_(std::move(reduce_buckets)),
          ds_ids_(std::move(ds_ids)),
          nccl_comm_(nccl_comm),
          comm_stream_(comm_stream)
    {
        for (long ds_id : ds_ids_) {
            has_acc_grad_[ds_id] = false;
            ag_comm_done_events_[ds_id] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
            ag_comp_done_events_[ds_id] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
            rs_comp_done_events_[ds_id] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
        }
    }
    ~CustomOpExecutor() {}

    void register_op_n_args(const std::string& op_name, long n_args, bool is_backward)
    {
        GraphOpStates& op_states = is_backward ? op_states_bwd_ : op_states_fwd_;
        op_states.registerOpNArgs(op_name, n_args);
    }

    void start_forward()
    {
        release_counter_.clear();
        op_states_fwd_.resetArgCounter();
    }

    void end_forward() {}

    void start_backward(bool update)
    {
        release_counter_.clear();
        op_states_bwd_.resetArgCounter();
        reduce_counter_ = ds_ids_.size();

        param_updated_ = update;
    }

    void end_backward()
    {
        if (param_updated_) {
            for (auto& it : has_acc_grad_) { it.second = false; }
        }
    }

    at::Tensor allgather_param(at::Tensor param_tensor, long ds_id)
    {
        const DSParam& param = param_registry_->getParam(ds_id);

        if (param_registry_->hasGatheredParam(ds_id)) {
            return param_registry_->getGatheredParam(ds_id);
        }

        const at::Tensor& ds_tensor = param.getDSTensor();
        at::Tensor output_buf = torch::empty(param.getShape(), ds_tensor.options());

        ag_comp_done_events_[ds_id]->record();
        ag_comp_done_events_[ds_id]->block(comm_stream_);
        ncclResult_t result = ncclAllGather(ds_tensor.contiguous().data_ptr(),
                                            output_buf.data_ptr(),
                                            ds_tensor.numel(),
                                            get_nccl_data_type(ds_tensor.scalar_type()),
                                            nccl_comm_,
                                            comm_stream_);

        if (result != ncclSuccess) { throw std::runtime_error("NCCL AllGather failed"); }

        ag_comm_done_events_[ds_id]->record(comm_stream_);

        param_registry_->registerGatheredParam(ds_id, output_buf);

        return output_buf;
    }

    at::Tensor allgather_param_symm_mem(
        at::Tensor param_tensor,
        long ds_id,
        c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> symm_mem)
    {
        const DSParam& param = param_registry_->getParam(ds_id);

        if (param_registry_->hasGatheredParam(ds_id)) {
            return param_registry_->getGatheredParam(ds_id);
        }

        const at::Tensor& ds_tensor = param.getDSTensor();
        at::Tensor output_buf = torch::empty(param.getShape(), ds_tensor.options());

        ag_comp_done_events_[ds_id]->record();
        ag_comp_done_events_[ds_id]->block(comm_stream_);

        {
            at::cuda::CUDAStreamGuard guard(comm_stream_);
            int world_size = process_group_->getSize();
            int rank = process_group_->getRank();

            at::Tensor local_buf =
                symm_mem->get_buffer(rank, ds_tensor.sizes(), ds_tensor.scalar_type(), 0);
            local_buf.copy_(ds_tensor, true);

            symm_mem->barrier(0);
            auto chunks = output_buf.flatten().chunk(world_size);
            for (int step = 0; step < world_size; step++) {
                int remote_rank = (rank - step + world_size) % world_size;
                auto src_buf = symm_mem->get_buffer(
                    remote_rank, ds_tensor.sizes(), ds_tensor.scalar_type(), 0);
                chunks[remote_rank].copy_(src_buf.flatten(), true);
            }
            symm_mem->barrier(0);
        }

        ag_comm_done_events_[ds_id]->record(comm_stream_);

        param_registry_->registerGatheredParam(ds_id, output_buf);

        return output_buf;
    }

    at::Tensor release_param(at::Tensor v, long ds_id, long count)
    {
        const DSParam& param = param_registry_->getParam(ds_id);

        if (!hasKey(release_counter_, ds_id)) { release_counter_[ds_id] = count; }
        release_counter_[ds_id]--;

        if (!param.isPersistent() && release_counter_[ds_id] == 0) {
            at::Tensor gathered_param = param_registry_->getGatheredParam(ds_id);
            const auto options = gathered_param.options();
            at::Tensor empty_buffer = torch::empty({0}, options);
            gathered_param.set_data(empty_buffer);

            param_registry_->unregisterGatheredParam(ds_id);
        }

        return v;
    }

    at::Tensor wait_allgather(at::Tensor v,
                              long ds_id,
                              const std::string& user,
                              long n_args,
                              bool is_backward)
    {
        GraphOpStates& op_states = is_backward ? op_states_bwd_ : op_states_fwd_;

        op_states.decrementArgCounter(user);

        if (op_states.isArgCounterZero(user)) {
            assert(hasKey(ag_comm_done_events_, ds_id));
            ag_comm_done_events_[ds_id]->block(at::cuda::getCurrentCUDAStream());
        }

        return v;
    }

    at::Tensor reduce_grad(at::Tensor grad_tensor, long ds_id)
    {
        int world_size = process_group_->getSize();
        const DSParam& param = param_registry_->getParam(ds_id);
        const auto scalar_type = grad_tensor.scalar_type();
        std::shared_ptr<ReduceBucket> reduce_bucket = reduce_buckets_->getBuffer(scalar_type);

        auto comp_stream = at::cuda::getCurrentCUDAStream();

        if (reduce_bucket->shouldFlush(grad_tensor.numel())) {
            flushReduceBucket(scalar_type);

            // reduce_bucket might be swapped in flushReduceBucket.
            reduce_bucket = reduce_buckets_->getBuffer(scalar_type);
        }

        if (grad_tensor.numel() > reduce_bucket->getSize()) {
            // extend buckets
            at::cuda::stream_synchronize(comm_stream_);
            reduce_bucket->reserve(grad_tensor.numel());
        }

        at::Tensor reduce_in_buffer = reduce_bucket->allocate(grad_tensor.numel());

        reduce_buckets_->getEvent(scalar_type)->block(comp_stream);
        reduce_in_buffer.copy_(grad_tensor.contiguous().view({-1}));
        reduce_tasks_[scalar_type].emplace_back(ds_id, reduce_in_buffer);

        rs_comp_done_events_[ds_id]->record(comp_stream);

        reduce_counter_--;

        if (reduce_counter_ == 0) {
            flushReduceBucket(scalar_type);

            // This looks duplicated but it's necessary to ensure the copy is done.
            // The backward hook has start_backward() might not be called when backward starting.
            reduce_counter_ = ds_ids_.size();

            // This synchronization ensures all of reduce calls are done before optimizer's step.
            at::cuda::stream_synchronize(comm_stream_);

            end_backward();
        }

        return at::Tensor();
    }

private:
    c10::intrusive_ptr<c10d::ProcessGroup> process_group_;
    std::shared_ptr<DSParamRegistry> param_registry_;
    std::shared_ptr<DoubleBufferedReduceBucket> reduce_buckets_;
    std::vector<long> ds_ids_;
    ncclComm_t nccl_comm_;
    at::cuda::CUDAStream comm_stream_;
    GraphOpStates op_states_fwd_ = GraphOpStates();
    GraphOpStates op_states_bwd_ = GraphOpStates();

    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> ag_comp_done_events_;
    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> ag_comm_done_events_;
    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> rs_comp_done_events_;

    std::unordered_map<long, size_t> release_counter_;
    size_t reduce_counter_ = 0;
    bool param_updated_ = false;
    std::unordered_map<at::ScalarType, std::vector<ReduceTask>> reduce_tasks_;
    std::unordered_map<long, bool> has_acc_grad_;

    void flushReduceBucket(at::ScalarType scalar_type)
    {
        if (!hasKey(reduce_tasks_, scalar_type)) { return; }

        int64_t tmp_recv_numel = 0;
        for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
            auto comp_done_event = rs_comp_done_events_.at(t.getDSId());
            comp_done_event->block(comm_stream_);

            if (has_acc_grad_.at(t.getDSId())) {
                tmp_recv_numel += param_registry_->getParam(t.getDSId()).getGradBuffer().numel();
            }
        }

        at::Tensor tmp_recv_buf = at::Tensor();
        if (tmp_recv_numel > 0) {
            tmp_recv_buf = torch::empty({tmp_recv_numel},
                                        at::TensorOptions().dtype(scalar_type).device(at::kCUDA));
        }

        ncclGroupStart();
        int64_t offset = 0;
        for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
            auto recv_buf = param_registry_->getParam(t.getDSId()).getGradBuffer();

            bool acc_grad = has_acc_grad_.at(t.getDSId());

            if (acc_grad) {
                recv_buf =
                    tmp_recv_buf.index({torch::indexing::Slice(offset, offset + recv_buf.numel())});
            }

            ncclResult_t result = ncclReduceScatter(t.getSendBuf().data_ptr(),
                                                    recv_buf.data_ptr(),
                                                    recv_buf.numel(),
                                                    get_nccl_data_type(scalar_type),
                                                    ncclAvg,
                                                    nccl_comm_,
                                                    comm_stream_);
            if (result != ncclSuccess) { throw std::runtime_error("NCCL ReduceScatter failed"); }

            if (acc_grad) { offset += recv_buf.numel(); }
        }
        ncclGroupEnd();

        {
            at::cuda::CUDAStreamGuard guard(comm_stream_);
            int64_t offset = 0;
            for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
                bool acc_grad = has_acc_grad_.at(t.getDSId());
                auto current_grad = param_registry_->getParam(t.getDSId()).getGradBuffer();

                if (acc_grad) {
                    auto recv_buf = param_registry_->getParam(t.getDSId()).getGradBuffer();
                    recv_buf.add_(tmp_recv_buf.index(
                        {torch::indexing::Slice(offset, offset + recv_buf.numel())}));
                    offset += recv_buf.numel();
                }
                has_acc_grad_[t.getDSId()] = true;
            }
        }

        reduce_buckets_->swap(scalar_type, comm_stream_);
        reduce_tasks_[scalar_type].clear();
    }
};

static std::shared_ptr<DSParamRegistry> param_registry;
static std::unordered_map<long, std::shared_ptr<CustomOpExecutor>> executors_;
std::shared_ptr<DoubleBufferedReduceBucket> reduce_buckets;
c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> symm_mem;

static at::cuda::CUDAStream comm_stream = at::cuda::getStreamFromPool(false);
static ncclComm_t nccl_comm;
static bool enable_double_buffer = false;
static bool use_symm_mem;
static bool profile = false;

std::vector<int64_t> sizes_to_int_vector(at::IntArrayRef sizes)
{
    std::vector<int64_t> result;
    for (int i = 0; i < sizes.size(); i++) { result.push_back(sizes[i]); }
    return result;
}

void lazy_init_symm_memory()
{
    if (use_symm_mem && !symm_mem) {
        int64_t max_param_size = 0;
        for (const auto& it : param_registry->getParams()) {
            int64_t size = it.second.getDSTensor().numel() * it.second.getDSTensor().element_size();
            if (size > max_param_size) { max_param_size = size; }
        }
        symm_mem = getSymmMemWorkspace(max_param_size);
    }
}

void enable_profiling(bool enable) { profile = enable; }

void register_graph(long graph_id, const std::vector<long>& ds_ids)
{
    executors_[graph_id] = std::make_shared<CustomOpExecutor>(
        process_group, param_registry, reduce_buckets, ds_ids, nccl_comm, comm_stream);
}

void register_graph_ops(long graph_id,
                        const std::vector<std::string>& op_names,
                        const std::vector<long>& n_args)
{
    assert(op_names.size() == n_args.size());
    for (int i = 0; i < op_names.size(); i++) {
        executors_[graph_id]->register_op_n_args(op_names[i], n_args[i], false);
    }
}

void register_bwd_graph_ops(long graph_id,
                            const std::vector<std::string>& op_names,
                            const std::vector<long>& n_args)
{
    assert(hasKey(executors_, graph_id));
    for (int i = 0; i < op_names.size(); i++) {
        executors_[graph_id]->register_op_n_args(op_names[i], n_args[i], true);
    }
}

void init(c10::intrusive_ptr<c10d::ProcessGroup> pg,
          int64_t initial_reduce_bucket_size,
          bool _use_symm_mem)
{
    process_group = pg;

    ncclUniqueId ncclID;
    ncclGetUniqueId(&ncclID);

    // ProcessGroup doesn't have an API to get the CUDA stream for comm calls.
    // So we create a NCCL communicator and call NCCL APIs directly.
    auto vec = std::vector<uint8_t>(reinterpret_cast<uint8_t*>(&ncclID),
                                    reinterpret_cast<uint8_t*>(&ncclID) + NCCL_UNIQUE_ID_BYTES);
    auto device = torch::Device(torch::kCUDA);
    at::Tensor tensor = torch::from_blob(vec.data(), {static_cast<long>(vec.size())}, torch::kUInt8)
                            .to(torch::Device(torch::kCUDA));
    std::vector<at::Tensor> bcast_input = {tensor};

    process_group->broadcast(bcast_input, c10d::BroadcastOptions())->wait();

    // create a new nccl communicator
    std::memcpy(&ncclID, tensor.to(torch::Device(torch::kCPU)).data_ptr(), NCCL_UNIQUE_ID_BYTES);
    ncclCommInitRank(&nccl_comm, process_group->getSize(), ncclID, process_group->getRank());

    param_registry = std::make_shared<DSParamRegistry>();
    reduce_buckets = std::make_shared<DoubleBufferedReduceBucket>(initial_reduce_bucket_size,
                                                                  enable_double_buffer);
    use_symm_mem = _use_symm_mem;
}

void cleanup()
{
    ncclCommDestroy(nccl_comm);
    process_group = nullptr;
}

void register_param(long ds_id,
                    const std::vector<int64_t>& ds_shape,
                    at::Tensor ds_tensor,
                    at::Tensor grad_buffer,
                    bool persistent)
{
    param_registry->registerParam(ds_id, ds_shape, ds_tensor, grad_buffer, persistent);
}

at::Tensor allgather_param(at::Tensor param_tensor, long graph_id, long ds_id)
{
    if (use_symm_mem) {
        return executors_[graph_id]->allgather_param_symm_mem(param_tensor, ds_id, symm_mem);
    }
    return executors_[graph_id]->allgather_param(param_tensor, ds_id);
}

at::Tensor allgather_param_meta(at::Tensor param_tensor, long graph_id, long ds_id)
{
    const DSParam& param = param_registry->getParam(ds_id);
    auto options = param.getDSTensor().options().device(c10::kMeta);
    at::Tensor output_buf = torch::empty(param.getShape(), options);
    return output_buf;
}

at::Tensor release_param(at::Tensor v, long graph_id, long ds_id, long count)
{
    return executors_[graph_id]->release_param(v, ds_id, count);
}

at::Tensor release_param_meta(at::Tensor v, long graph_id, long ds_id, long count) { return v; }

at::Tensor wait_allgather(at::Tensor v,
                          long graph_id,
                          long ds_id,
                          const std::string& user,
                          long n_args,
                          bool is_backward)
{
    executors_[graph_id]->wait_allgather(v, ds_id, user, n_args, is_backward);
    return v;
}

at::Tensor wait_allgather_meta(at::Tensor v,
                               long graph_id,
                               long ds_id,
                               const std::string& user,
                               long n_args,
                               bool is_backward)
{
    return v;
}

at::Tensor reduce_grad(at::Tensor grad_tensor, long graph_id, long ds_id)
{
    if (!profile) { executors_[graph_id]->reduce_grad(grad_tensor, ds_id); }
    return at::Tensor();
}

at::Tensor reduce_grad_meta(at::Tensor grad_tensor, long graph_id, long ds_id)
{
    return at::Tensor();
}

void start_forward()
{
    lazy_init_symm_memory();
    for (auto& it : executors_) { it.second->start_forward(); }
}

void end_forward()
{
    for (auto& it : executors_) { it.second->end_forward(); }
}

void start_backward(bool update)
{
    for (auto& it : executors_) { it.second->start_backward(update); }
}

void end_backward()
{
    for (auto& it : executors_) { it.second->end_backward(); }
}

at::Tensor test_call(at::Tensor a)
{
    std::cout << "test_call" << std::endl;
    return a;
}

}  // namespace n3z

TORCH_LIBRARY(native_z3, m)
{
    m.def("allgather_param(Tensor a, int graph_id, int id) -> Tensor");
    m.def("release_param(Tensor a, int graph_id, int id, int count) -> Tensor");
    m.def(
        "wait_allgather(Tensor a, int graph_id, int id, str user, int n_args, bool bwd) -> Tensor");
    m.def("reduce_grad(Tensor a, int graph_id, int id) -> Tensor");

    m.def("test_call(Tensor a) -> Tensor");
}

TORCH_LIBRARY_IMPL(native_z3, CPU, m)
{
    m.impl("allgather_param", &n3z::allgather_param);
    m.impl("release_param", &n3z::release_param);
    m.impl("wait_allgather", &n3z::wait_allgather);
    m.impl("reduce_grad", &n3z::reduce_grad);

    m.impl("test_call", &n3z::test_call);
}

TORCH_LIBRARY_IMPL(native_z3, CUDA, m)
{
    m.impl("allgather_param", &n3z::allgather_param);
    m.impl("release_param", &n3z::release_param);
    m.impl("wait_allgather", &n3z::wait_allgather);
    m.impl("reduce_grad", &n3z::reduce_grad);

    m.impl("test_call", &n3z::test_call);
}

TORCH_LIBRARY_IMPL(native_z3, Meta, m)
{
    m.impl("allgather_param", &n3z::allgather_param_meta);
    m.impl("release_param", &n3z::release_param_meta);
    m.impl("wait_allgather", &n3z::wait_allgather_meta);
    m.impl("reduce_grad", &n3z::reduce_grad_meta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("register_param", &n3z::register_param, "Register a parameter");
    m.def("enable_profiling", &n3z::enable_profiling, "Enable profiling");
    m.def("init", &n3z::init, "Set the process group");
    m.def("cleanup", &n3z::cleanup, "Cleanup the process group");
    m.def("register_graph", &n3z::register_graph, "Register graph with a list of ds parameter ids");
    m.def("register_graph_ops",
          &n3z::register_graph_ops,
          "Register the number of arguments for an op");
    m.def("register_bwd_graph_ops",
          &n3z::register_bwd_graph_ops,
          "Register the number of arguments for a backward op");
    m.def("start_forward", &n3z::start_forward, "Start forward pass");
    m.def("end_forward", &n3z::end_forward, "End forward pass");
    m.def("start_backward", &n3z::start_backward, "Start backward pass");
    m.def("end_backward", &n3z::end_backward, "End backward pass");
}
