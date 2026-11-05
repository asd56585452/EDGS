import re

with open('/home/tsaichenghan/EDGS/source/trainer.py', 'r') as f:
    content = f.read()

pattern = r'    def save_model\(self\):.*?(?=    def init_with_corr)'

lines = [
    '    def save_model(self):',
    '        print(f"[ITER {self.gs_step}] Saving Gaussians & Standard Universal PLY")',
    '        self.scene.save(self.gs_step)',
    '        root_ply_path = os.path.join(self.scene.model_path, f"point_cloud_step_{self.gs_step}.ply")',
    '        self.GS.gaussians.save_ply(root_ply_path)',
    '        print(f"[ITER {self.gs_step}] Saving Checkpoint: {root_ply_path}")',
    '        try:',
    '            import numpy as np',
    '            xyz = self.GS.gaussians.get_xyz.detach().cpu().numpy()',
    '            f_dc = self.GS.gaussians._features_dc.detach().cpu().numpy().reshape(-1, 3)',
    '            rgb = np.clip((f_dc * 0.28209479177387814 + 0.5) * 255.0, 0, 255).astype(np.uint8)',
    '            header_lines = [',
    '                "ply",',
    '                "format binary_little_endian 1.0",',
    '                f"element vertex {len(xyz)}",',
    '                "property float x",',
    '                "property float y",',
    '                "property float z",',
    '                "property uchar red",',
    '                "property uchar green",',
    '                "property uchar blue",',
    '                "end_header"',
    '            ]',
    '            header_bytes = "\\n".join(header_lines).encode("ascii") + b"\\n"',
    '            out_data = np.empty(len(xyz), dtype=[',
    '                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),',
    '                ("red", "u1"), ("green", "u1"), ("blue", "u1")',
    '            ])',
    '            out_data["x"] = xyz[:, 0]',
    '            out_data["y"] = xyz[:, 1]',
    '            out_data["z"] = xyz[:, 2]',
    '            out_data["red"] = rgb[:, 0]',
    '            out_data["green"] = rgb[:, 1]',
    '            out_data["blue"] = rgb[:, 2]',
    '            rgb_ply_path = os.path.join(self.scene.model_path, f"point_cloud_step_{self.gs_step}_rgb.ply")',
    '            with open(rgb_ply_path, "wb") as f_rgb:',
    '                f_rgb.write(header_bytes)',
    '                f_rgb.write(out_data.tobytes())',
    '            print(f"[ITER {self.gs_step}] Saving MeshLab RGB PLY: {rgb_ply_path}")',
    '        except Exception as e:',
    '            print(f"Warning: failed to save RGB ply: {e}")',
    '        torch.save((self.GS.gaussians.capture(), self.gs_step),',
    '                self.scene.model_path + "/chkpnt" + str(self.gs_step) + ".pth")',
    '',
    ''
]
replacement = chr(10).join(lines)

new_content = re.sub(pattern, lambda m: replacement, content, flags=re.DOTALL)
with open('/home/tsaichenghan/EDGS/source/trainer.py', 'w') as f:
    f.write(new_content)

print("Applied fix_save_model successfully!")
