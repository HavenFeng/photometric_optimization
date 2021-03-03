from yacs.config import CfgNode
import os

cfg = CfgNode()

abs_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg.root_dir = abs_root_dir
cfg.device = 'cuda'
cfg.device_id = '0'
cfg.face_detect_type = "2D"
cfg.flame_model_path = os.path.join(cfg.root_dir, 'model', 'generic_model.pkl')
cfg.tex_space_path = os.path.join(cfg.root_dir, 'model', 'FLAME_texture.npz')
cfg.flame_model_path = os.path.join(cfg.root_dir, 'model', 'generic_model.pkl')
cfg.rect_model_path = os.path.join(cfg.root_dir, 'model', 's3fd.pth')
cfg.face_seg_model = os.path.join(cfg.root_dir, 'model', 'face_seg.pth')
cfg.landmark_model_path = os.path.join(cfg.root_dir, 'model', '2DFAN4-11f355bf06.pth.tar')
cfg.flame_lmk_embedding_path = os.path.join(cfg.root_dir, 'data', 'landmark_embedding.npy')
cfg.mesh_file = os.path.join(cfg.root_dir, 'data', 'head_template_mesh.obj')
cfg.save_folder = os.path.join(cfg.root_dir, 'test_results')

cfg.camera_params = 3
cfg.shape_params = 100
cfg.expression_params = 50
cfg.pose_params = 6
cfg.tex_params = 50
cfg.seg_class = 19
cfg.use_face_contour = True
cfg.cropped_size = 256
cfg.batch_size = 1
cfg.image_size = 224
cfg.rect_thresh = 0.5
cfg.e_lr = 0.005
cfg.e_wd = 0.0001
cfg.w_pho = 8
cfg.w_lmks = 1
cfg.max_iter = 2000
cfg.w_shape_reg = 1e-4
cfg.w_expr_reg = 1e-4
cfg.w_pose_reg = 0


