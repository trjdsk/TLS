#include <cstdint>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <algorithm>

#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "edge-impulse-sdk/classifier/ei_classifier_types.h"
#include "edge-impulse-sdk/dsp/numpy.hpp"
#include "edge-impulse-sdk/porting/ei_classifier_porting.h"
#include "model-parameters/model_metadata.h"
#include "model-parameters/model_variables.h"

extern "C" {

// Global impulse
static ei_impulse_handle_t* g_impulse_handle = nullptr;

int ei_model_init() {
    g_impulse_handle = new ei_impulse_handle_t(&impulse);
    if (!g_impulse_handle) return -1;
    EI_IMPULSE_ERROR res = init_impulse(g_impulse_handle);
    return (res == EI_IMPULSE_OK) ? 0 : -1;
}

void ei_model_cleanup() {
    if (g_impulse_handle) {
        delete g_impulse_handle;
        g_impulse_handle = nullptr;
    }
}

int ei_model_is_initialized() {
    return g_impulse_handle ? 1 : 0;
}

int ei_model_get_input_shape(int* w, int* h, int* c) {
    if (!w || !h || !c) return -1;
    *w = EI_CLASSIFIER_INPUT_WIDTH;
    *h = EI_CLASSIFIER_INPUT_HEIGHT;
    *c = 1; // grayscale
    return 0;
}

int ei_model_get_output_dims(int* n) {
    if (!n) return -1;
    *n = EI_CLASSIFIER_LABEL_COUNT;
    return 0;
}

// Inference: grayscale 96x96 uint8
int ei_model_infer(const uint8_t* image_data, float* output_scores) {
    if (!g_impulse_handle || !image_data || !output_scores) return -1;

    const size_t data_size = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
    std::vector<float> float_data(data_size);

    for (size_t i = 0; i < data_size; i++)
        float_data[i] = static_cast<float>(image_data[i]) / 255.0f;

    signal_t signal;
    int err = numpy::signal_from_buffer(float_data.data(), data_size, &signal);
    if (err != 0) return -1;

    ei_impulse_result_t result;
    EI_IMPULSE_ERROR r = process_impulse(g_impulse_handle, &signal, &result, false);
    if (r != EI_IMPULSE_OK) return -1;

    for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++)
        output_scores[i] = result.classification[i].value;

    return 0;
}
}
