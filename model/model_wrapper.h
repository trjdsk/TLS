/*
 * Edge Impulse Model Wrapper Header for Python Integration
 * 
 * This header defines the C interface for the Edge Impulse model wrapper
 * to enable Python integration via ctypes.
 */

#ifndef MODEL_WRAPPER_H
#define MODEL_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the Edge Impulse model
 * @return 0 on success, -1 on failure
 */
int ei_model_init();

/**
 * Cleanup the Edge Impulse model
 */
void ei_model_cleanup();

/**
 * Get model input dimensions
 * @param width Output width
 * @param height Output height
 * @param channels Output channels
 * @return 0 on success, -1 on failure
 */
int ei_model_get_input_dims(int* width, int* height, int* channels);

/**
 * Get model output dimensions
 * @param num_classes Output number of classes
 * @return 0 on success, -1 on failure
 */
int ei_model_get_output_dims(int* num_classes);

/**
 * Run inference on a grayscale image
 * @param image_data Pointer to grayscale image data (96x96 pixels)
 * @param output_scores Pointer to output array for class scores (2 elements)
 * @return 0 on success, -1 on failure
 */
int ei_model_infer(const unsigned char* image_data, float* output_scores);

/**
 * Run inference and get embedding features (for palm recognition)
 * @param image_data Pointer to grayscale image data (96x96 pixels)
 * @param embedding_data Pointer to output array for embedding features
 * @param embedding_size Size of embedding array
 * @return 0 on success, -1 on failure
 */
int ei_model_get_embedding(const unsigned char* image_data, float* embedding_data, int embedding_size);

/**
 * Get class labels
 * @param labels Pointer to array of strings
 * @param num_labels Number of labels
 * @return 0 on success, -1 on failure
 */
int ei_model_get_labels(const char** labels, int* num_labels);

/**
 * Check if model is initialized
 * @return 1 if initialized, 0 if not
 */
int ei_model_is_initialized();

#ifdef __cplusplus
}
#endif

#endif // MODEL_WRAPPER_H
