# Images Needed for Optimization Lab

This folder should contain the following images referenced in the notebook:

## Required Images

1. **cost.jpg**
   - Description: A 3D visualization showing a hilly landscape representing a cost function
   - Purpose: Illustrates the concept of "going downhill" to minimize cost
   - Dimensions: ~650x300 pixels
   - Shows: Gradient descent path on a cost surface

2. **kiank_sgd.png**
   - Description: Comparison of SGD vs GD convergence paths
   - Purpose: Shows oscillations in SGD vs smooth convergence in GD
   - Dimensions: ~750x250 pixels
   - Shows: Two plots with "+" marking the minimum, demonstrating different convergence patterns

3. **kiank_minibatch.png**
   - Description: Comparison of SGD vs Mini-Batch GD
   - Purpose: Shows that mini-batch GD leads to faster optimization than pure SGD
   - Dimensions: ~750x250 pixels
   - Shows: Convergence paths with "+" marking the minimum

4. **kiank_shuffle.png**
   - Description: Visualization of the shuffling process
   - Purpose: Shows how training examples are randomly shuffled
   - Dimensions: ~550x300 pixels
   - Shows: Before and after shuffling of data columns (X and Y synchronized)

5. **kiank_partition.png**
   - Description: Visualization of partitioning shuffled data into mini-batches
   - Purpose: Shows how shuffled data is split into fixed-size mini-batches
   - Dimensions: ~550x300 pixels
   - Shows: Data divided into 64-example mini-batches with potentially smaller last batch

6. **opt_momentum.png**
   - Description: Visualization of momentum optimization
   - Purpose: Shows how momentum smooths the gradient descent path
   - Dimensions: ~400x250 pixels
   - Shows: Red arrows for momentum steps, blue points for gradients

7. **lr.png**
   - Description: Learning rate decay schedule visualization
   - Purpose: Shows how learning rate decreases over epochs with fixed interval scheduling
   - Dimensions: ~400x250 pixels
   - Shows: Step-wise learning rate decay over training epochs

## Notes

- All images should be in PNG or JPG format
- Images are referenced in the notebook using relative paths: `images/filename.ext`
- If images are missing, the notebook will still run but visualization cells will show broken image links
- You can create placeholder images or use diagrams that illustrate the concepts described above
