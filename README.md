# Co-occurrence2Dand3D
Stores Python codes for calculation of extended multi-sort co-occurrence matrices for 2D and 3D image analysis.

Extended multi-sort IIGGAD co-occurrence matrices count occurrences of pairs of pixels in the target
image which have certain properties. The matrix considers both pixels' intensity (I), gradient magnitudes (G),
angle between gradient directions (A) and distance between the two pixels (D).
Extended IIID co-occurrence matrices count occurrences of triplets of pixels.

Relevant publications:
> V. A. Kovalev, F. Kruggel, H.-J. Gertz and D. Y. von Cramon, 
"Three-dimensional texture analysis of MRI brain datasets," 
in IEEE Transactions on Medical Imaging, vol. 20, no. 5, pp. 424-433, May 2001.
doi: 10.1109/42.925295

> Kovalev V., Dmitruk A., Safonau I., Frydman M., Shelkovich S. (2011) 
A Method for Identification and Visualization of Histological Image 
Structures Relevant to the Cancer Patient Conditions. 
In: Real P., Diaz-Pernil D., Molina-Abril H., Berciano A., Kropatsch W. (eds) 
Computer Analysis of Images and Patterns. CAIP 2011. Lecture Notes in 
Computer Science, vol 6854. Springer, Berlin, Heidelberg

Functions to use:
* `cooccur2D` for calculation of IIGGAD matrices and their sub-variants
for 2D gray-level images, only pixel pairs are considered.
* `cooccur2Dn` for calculation of IID and IIID matrices
for 2D gray-level images, pixel pairs and triplets are considered.
* `cooccur3D` (*coming soon*) for calculation of IIGGAD matrices and their sub-variants
for 3D gray-level images, only pixel pairs are considered.

GGAD and AD sub-variants of the multi-sort matrices can be used to 
analyse image anisotropy.

See **run_example_...py** files for use cases.    

### Examples

* An ordinary image. IID matrix with 6 intensity bins, D=1.

 ![Alt text](readme_figs/01_Lena_IID.png?raw=true "Title")

* Same image. AD matrix with D=1 (histogram of angles at adjacent pixels)

 ![Alt text](readme_figs/02_Lena_AD.png?raw=true "Title")
 
 * Same image, but only Region of Interest (ROI) is processed.  

 ![Alt text](readme_figs/03_Lena_AD_roi1.png?raw=true "Title")
 
 * Same image, different ROI.  

 ![Alt text](readme_figs/04_Lena_AD_roi2.png?raw=true "Title")
 
 * Noisy image, AD matrix. Gradients in neighbouring pixels are oriented differently.  

 ![Alt text](readme_figs/05_noise_AD.png?raw=true "Title")

 * Grass, AD matrix. Gradient orientation in neighbouring pixels are more correlated.  

 ![Alt text](readme_figs/06_grass_AD.png?raw=true "Title")

* Grass, AD matrix, D=10. Correlation between gradient orientations drops with increase of distance between pixels.  

 ![Alt text](readme_figs/07_grass_AD_dist10.png?raw=true "Title")
