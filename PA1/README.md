# Optimization using Least Square

## Analysis

The problem should be image segmentation by optimizing cost function and scribbles image using least square.

$$J\left ( L_{i} \right )= \left ( \hat{L_{i}}(r)-\sum_{s\in N(r)}^{}w_{rs}\cdot \hat{L_{i}}(s) \right )^{2}$$

- $L_{i}$ : scribbles label
- $\hat{L_{i}}$ : expected label

The cost function of this problem is defined by **Equation 1** To obtain the cost function, we first obtain a neighborhood matrix composed of the neighborhood weight of each pixel. Then optimize the calculated cost function and scribble image. At this time, we should use the least square.

![image](https://user-images.githubusercontent.com/125437452/221482234-e3d04678-9310-4b09-85ec-b6cfb6d131b4.png)


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

In this experiment, 3×3 kenerl was used, and the weight function used is as follows.


|wrs=Yr-Y(s)|<p>**Equation 2.**</p><p>Original weight function</p>|
| :-: | :-: |
|wrs∝e-Yr-Y(s)22σr2|<p>**Equation 3.**</p><p>Option1 weight function</p>|

1. **Binary-label semantic segmentation**

|![](Aspose.Words.d327dd48-e01e-4293-9220-b0c6f3bf8883.002.png)|![](Aspose.Words.d327dd48-e01e-4293-9220-b0c6f3bf8883.003.png)|
| :- | :- |
**Figure 2.** Binary-label sematic segmentation result.

Left image is used original weight function and right image is used option1 weight function.


|***Class***|***Original IOU***|***Option1 IOU***|
| :-: | :-: | :-: |
|*Background*|0.896986834|0.972299457|
|*Foreground*|0.987832536|0.953637905|
|*mIOU*|0.942409685|**0.962968681**|

1. **Multi-label semantic segmentation**

|![](Aspose.Words.d327dd48-e01e-4293-9220-b0c6f3bf8883.004.png)|
| :-: |
**Figure 3.** Multi-label sematic segmentation result..


|***Class***|***Original IOU***|
| :-: | :-: |
|*Sky(green)*|0.785511|
|*Buildings(yellow)*|0.907131|
|*Tree(blue)*|0.917435|
|*Hair(sky blue)*|0.917713|
|*Skin(white)*|0.856642|
|*Phone(pink)*|0.892648|
|*Clothes(red)*|0.972529|
|*mIOU*|**0.892801**|

**Discussion**

The experiment was conducted using various kernels, but overflow occurred and the result could not be output. In addition, the problem was executed by the CPU and GPU. Although it showed a better speed in the GPU environment with the same code, multi-layer could not output the result due to an error in the library used. However, with the same code, the CPU output the result of multi-layer. While performing this task, I felt that the results could vary depending on what library is used and what environment it is.
20214029 서나미

**Code implementation.**

The entire code file consists of main.py, function.py, and last\_square.py, and the execution process is as follows.

- Load data
- Calculate neighborhood weights
- Calculate least square solution
- Calculate IOU

1. **Load data**

**def** load\_classifier(name): 

`	`**return** classifier, class\_name

Read “class\_names.txt” and **return** the name of class and the number of class.



1. **Calculate neighborhood weights**

**def** calculate\_neighborhood\_weight(name, image, scribbles, kernel\_size, classifier):

`	`**return** cost\_mat, lables

Calculate\_neighborhood\_weight function calculates weight matrix and cost matrix at the same time, and finally returns cost matrix. The reason for this implementation can reduce the computation time compared to when calculating cost matrix after calculating weight matrix.


The implementation order of the Calculate\_neighborhood\_weight function is as follows.

- Compute neighborhood means and variations
- Compute weights and cost function


1. **Compute neighborhood means and variations**

**def** compute\_statics(image, kernel\_size): 

`	`**return** neighbor\_mean, neighbor\_variation

The compute\_statics function calculates the neighborhood mean of the image through a convolution operation. A kernel size of convolution is a kernel size of neighborhood size.

![](Aspose.Words.d327dd48-e01e-4293-9220-b0c6f3bf8883.005.png)

**Figure 4.** Compute neighborhood sum and the number of neighborhoods

When a kernel is configured and convolution is performed as shown in **Figure 4**, the sum of intensity of neighborhood pixels is obtained. Likewise, performing convolution with an image with all pixel values of 1 can obtain the number of neighbors. Using this method, edge processing can be simplified and calculation speed can be improved.



1. **Compute weights and cost function**

**def** original(neighbor\_pixels, r, variation, mean):

**def** option1(neighbor\_pixels, r, variation, mean):

Each function is implemented using **equations 2** and **3** and finally calculated for cost function. It is stored in a sparse.csc\_matrix and returns its value.



1. **Calculate least square solution**

**def** calculate\_least\_square(weight\_mat, lables):

`	`**return** res

The calculate\_least\_square function optimizes cost matrix and labels to return the result.



1. **Calculate IOU**

**def** calculate\_IOU(name, classfier, res, flag):

`	`**return** iou\_lables, m\_iou

A∩BA∪B

**Equation 4** IOU

IOU is calculated using **Equation 4**.

![텍스트이(가) 표시된 사진

자동 생성된 설명](Aspose.Words.d327dd48-e01e-4293-9220-b0c6f3bf8883.006.png)

**Figure 5.** Calculate the intersection

The calculate\_IOU function calculates the intersection by taking and operation calculating the ground truth image and the result image.

**SAVE**

The res\_img.png file is a labeled file, and seg\_img.png is an image quantized with label. The IOU of each test is stored in the iou.csv file.

**Comment**

![텍스트이(가) 표시된 사진

자동 생성된 설명](Aspose.Words.d327dd48-e01e-4293-9220-b0c6f3bf8883.007.png)

**Figure 6.** Error

The experiment was conducted on the GPU and CPU. Binary-label was performed in the GPU environment, and multi-label was performed in the CPU environment. When multi-label was performed in a GPU environment, an error such as Figure 6 occurred. Eventually, the experiment was conducted in a CPU environment without solving it.
