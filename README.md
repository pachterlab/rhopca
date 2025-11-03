# rhoPCA


Implements generalized eigenvalue decomposition for target and background (or two contrasted conditions) scRNA-seq data. Written for compatibility with anndata objects. 

$\rho$ PCA finds axes that maximize variation in the target samples while minimizing variance in the background samples. 

![Schematic of the method](assets/schematic.png)

