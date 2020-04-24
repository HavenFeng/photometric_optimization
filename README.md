# Photometric FLAME Fitting

This repository provides an analysis-by-synthesis framework to fit a textured [FLAME](http://flame.is.tue.mpg.de/) model to an image. FLAME is a lightweight generic 3D head model learned from over 33,000 head scans, but it does not come with an appearance space (see the [scientific publication](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/400/paper.pdf) for details). 

This repository 
1) describes how to build a texture space for FLAME from in-the-wild images, and provides
2) code to fit a textured FLAME model to in-the-wild images, optimizing for FLAME's parameters, appearance, and lighting, and
3) code to optimize for the FLAME texture to match an in-the-wild image. 

**The codes and demos will be released soon.**

## Create FLAME texture space from in-the-wild images

**The FLAME texture space can be acquired from the [FLAME project website](https://flame.is.tue.mpg.de).**
<p align="left"> 
<img src="flame_texture.png" width="150"/>
</p>
<p align="left">Image from the [3DMM survey paper]() <p align="left">

Given that the widely used facial albedo space of [Basel Face Model(BFM)](https://faces.dmi.unibas.ch/bfm/index.php?nav=1-0&id=basel_face_model) is only built with 200 subjects, we want to build a texture space which covers a large range of ethnicity from in-the-wild data. Therefore, we pre-select 1500 images from [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) then use this repo to optmize the FLAME model to get the accurate 3D reconstruction, and further obtain the corresponding albedo textures(the illuminance is estimated and removed, the initial albedo is from BFM).\
The FLAME texture space is a PCA space of these 1500 albedo textures.

## Notes
We use the FLAME.py from [FLAME_PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch), the renderer.py is heavily adapted from [DECA](https://github.com/YadiraF/DECA)


## License
This code is available for non-commercial scientific research purposes as defined in the [LICENSE](https://github.com/YadiraF/DECA/blob/master/LICENSE) file.

## Contact
Please feel free to contact haiwen.feng@tuebingen.mpg.de for any related issue
