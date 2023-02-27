# Optimization using Least Square

## Analysis

The problem should be image segmentation by optimizing cost function and scribbles image using least square.

$$J\left ( L_{i} \right )= \left ( \hat{L_{i}}(r)-\sum_{s\in N(r)}^{}w_{rs}\cdot \hat{L_{i}}(s) \right )^{2}$$

- $L_{i}$ : scribbles label
- $\hat{L_{i}}$ : expected label

The cost function of this problem is defined by **Equation 1** To obtain the cost function, we first obtain a neighborhood matrix composed of the neighborhood weight of each pixel. Then optimize the calculated cost function and scribble image. At this time, we should use the least square.

![image](https://user-images.githubusercontent.com/125437452/221488985-b423b9c8-d892-4a76-96d2-5c8a17420c24.png)
**Figure 1.** Neighborhood matrix

The red square box in **Figure 1** is $r$-pixel, and the blue square box is kernel that determines the boundary of neighborhood. That is, the values in the blue square box become the neighborhood pixel of the $r$-pixel. In this case, the value of the neighborhood matrix is calculated as the absolute value of the difference between intensity of $s$ and intensity of $r$, or is calculated using various weight functions.

- $r$ : coordinate of center $r$ pixel
- $s$ : coordinate of center of $r$
- $N(r)$ : Neighborhood pixels of $r$
- Neighborhood weights ($w_{rs}$)

The order of the total tasks is as follows.

- Task 1. Calculate neighborhood weights
- Task 2. Calculate least square solution
- Task 3. Calculate IOU

## Result


### Binary-label semantic segmentation

![image](https://user-images.githubusercontent.com/125437452/221488490-270633df-5f84-4c51-ab44-295535253b9f.png)
**Figure 2.** Left image is used original weight function and right image is used option1 weight function.


|**Class**|**Original IOU**|**Option1 IOU**|
| :-: | :-: | :-: |
|*Background*|0.896986834|0.972299457|
|*Foreground*|0.987832536|0.953637905|
|*mIOU*|0.942409685|**0.962968681**|

### Multi-label semantic segmentation
![image](https://user-images.githubusercontent.com/125437452/221488774-7450ba27-8f5e-4bb6-b292-da5d18aab53b.png)

**Figure 3.** Multi-label sematic segmentation result..


|**Class**|**Original IOU**|
| :-: | :-: |
|*Sky(green)*|0.785511|
|*Buildings(yellow)*|0.907131|
|*Tree(blue)*|0.917435|
|*Hair(sky blue)*|0.917713|
|*Skin(white)*|0.856642|
|*Phone(pink)*|0.892648|
|*Clothes(red)*|0.972529|
|*mIOU*|**0.892801**|

## Discussion

The experiment was conducted using various kernels, but overflow occurred and the result could not be output. In addition, the problem was executed by the CPU and GPU. Although it showed a better speed in the GPU environment with the same code, multi-layer could not output the result due to an error in the library used. However, with the same code, the CPU output the result of multi-layer. While performing this task, I felt that the results could vary depending on what library is used and what environment it is.
