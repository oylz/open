#ifndef IDOL_DENOISER_PLUGIN_H
#define IDOL_DENOISER_PLUGIN_H

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nvinfer1 {

class idol_denoiser_plugin_creator final : public IPluginCreator {
public:
    idol_denoiser_plugin_creator();
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
private:
    std::string mNamespace;
    PluginFieldCollection mFC{};
};

struct IdolTensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> host;
    float* device = nullptr;
};

struct IdolConfig {
    int q_token_length = 8192;
    int in_channels = 16;
    int cam_channels = 5;
    int model_channels = 1024;
    int cond_channels = 1280;
    int cond2_channels = 128;
    int num_blocks = 24;
    int num_refiner_blocks = 2;
    int num_heads = 16;
    int num_head_channels = 64;
    float mlp_ratio = 4.0f;
};

class idol_denoiser_plugin final : public IPluginV2DynamicExt {
private:
    bool loadWeights(const std::string& path);
    void freeDeviceWeights();
    float* weight(const std::string& name) const;

public:
    const char* getPluginType() const noexcept override { return "IdolDenoiser"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    int getNbOutputs() const noexcept override { return 2; } // [0]: out_camera [B,1,5], [1]: out_latent [B,8192,16]
    void setPluginNamespace(const char* ns) noexcept override { mNamespace = ns ? ns : ""; }
    void destroy() noexcept override { delete this; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

    idol_denoiser_plugin();
    idol_denoiser_plugin(const void* serialData, size_t serialLength);
    ~idol_denoiser_plugin() override;

    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override;
    IPluginV2DynamicExt* clone() const noexcept override;
    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

    // Inputs: [0] camera [B,1,5], [1] latent [B,8192,16], [2] t [B], [3] feature1 [B,Lc,1280], [4] feature2 [B,Lc,128]
    // Outputs: [0] out_camera [B,1,5], [1] out_latent [B,8192,16]
    int enqueue(
        const PluginTensorDesc* inputDesc,
        const PluginTensorDesc* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream
    ) noexcept override;

private:
    IdolConfig mCfg{};
    std::unordered_map<std::string, IdolTensor> mWeights;
    bool mWeightsLoaded = false;
    std::string mNamespace;
    std::string mWeightPath = "./idol_denoiser_weights.bin";

    int mB = 2;
    int mLc = 4101;

    int run_denoiser(
        const float* camera,
        const float* latent,
        const float* t,
        const float* feature1,
        const float* feature2,
        float* out_camera,
        float* out_latent,
        cudaStream_t stream
    );
};

} // namespace nvinfer1

#endif





