# photometric_optimization
Photometric optimization in PyTorch for getting head pose, shape, expression, albedo and lighting. 
This repository includes
1. Optimize the FLAME model to an in-the-wild face image and corresponding 2D landmarks.
2. Optimize to get facial albedo textures with in-the-wild face images and corresponding 3D reconstructions.
The codes and demos will be released soon.

## Texture optimization on in-the-wild images
**We use this code to create the FLAME texture model, which can be acquired from the [FLAME project website](https://flame.is.tue.mpg.de).**\\
<p align="left"> 
<img src="test_images/results/teaser.gif">
</p>
<p align="left">Image from the [3DMM survey paper]() <p align="left">

Given that the widely used facial albedo model of [Basel Face Model(BFM)](https://faces.dmi.unibas.ch/bfm/index.php?nav=1-0&id=basel_face_model) is only built with 200 subjects, we want to build a texture model which covers a large range of ethnicity from in-the-wild data. Therefore, we pre-select 1500 images from [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) then use this code to optmize the FLAME model to get the accurate 3D correspondence and obtain the albedo textures(the illuminance is estimated and removed, the initial albedo is from BFM). The FLAME texture model is a PCA space of these 1500 albedo textures.

## Notes
We use the FLAME.py from [FLAME_PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch), the renderer part is heavily adapted from [DECA](https://github.com/YadiraF/DECA)


## License
This code and model are available for non-commercial scientific research purposes as defined in the [LICENSE](https://github.com/YadiraF/DECA/blob/master/LICENSE) file.

## Contact
Please feel free to contact haiwen.feng@tuebingen.mpg.de for any related issue
