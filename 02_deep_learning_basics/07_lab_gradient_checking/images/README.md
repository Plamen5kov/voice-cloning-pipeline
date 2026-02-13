# Images for Gradient Checking Lab

This directory should contain the following images used in the notebook:

## Required Images

### 1. `1Dgrad_kiank.png`
- **Description**: Diagram showing 1-dimensional gradient checking
- **Shows**: Forward propagation (x → J) and backward propagation (computing dJ/dθ)
- **Used in**: Section 4 - 1-Dimensional Gradient Checking
- **Dimensions**: Approximately 600x250 pixels

### 2. `NDgrad_kiank.png`
- **Description**: Diagram of the deep neural network architecture
- **Shows**: 3-layer network with LINEAR → RELU → LINEAR → RELU → LINEAR → SIGMOID
- **Used in**: Section 5 - N-Dimensional Gradient Checking
- **Dimensions**: Approximately 600x400 pixels

### 3. `dictionary_to_vector.png`
- **Description**: Visualization of parameter dictionary to vector conversion
- **Shows**: How parameters (W1, b1, W2, b2, W3, b3) are flattened and concatenated into a single vector
- **Shows**: Inverse operation (vector_to_dictionary) as well
- **Used in**: Section 5 - N-Dimensional Gradient Checking (before Exercise 4)
- **Dimensions**: Approximately 600x400 pixels

## Image Sources

These images are typically from:
- Deep Learning Specialization course materials (Coursera/deeplearning.ai)
- Credit: Andrew Ng and team

## If Images Are Missing

If you don't have these images, the notebook will still function but you'll see broken image links. The text descriptions in the markdown cells provide the necessary context to understand the concepts.

To add images:
1. Place the image files in this directory
2. Ensure filenames match exactly (case-sensitive)
3. Supported formats: PNG (recommended), JPG

## Creating Your Own Images (Optional)

If you need to create these images yourself, consider:
- **1Dgrad_kiank.png**: Simple flowchart showing x → [multiply by θ] → J, with derivative arrow going back
- **NDgrad_kiank.png**: Network diagram with 3 layers showing dimensions (4→5→3→1)
- **dictionary_to_vector.png**: Diagram showing dictionary with W1, b1, etc. being flattened into column vector
