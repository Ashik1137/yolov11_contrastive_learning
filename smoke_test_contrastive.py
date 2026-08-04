import torch

from ultralytics.nn.tasks import DetectionModel

model = DetectionModel("yolo11s.yaml", nc=1, ch=3, verbose=False)
model.args = type("Args", (), {"contrastive": True, "temperature": 0.5, "contrastive_weight": 0.1})()
model.eval()
model.init_contrastive_modules()

x = torch.rand(2, 3, 64, 64)
batch_idx = torch.tensor([0, 0, 1], dtype=torch.long)
bboxes = torch.tensor([[0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.2, 0.2], [0.4, 0.4, 0.2, 0.2]], dtype=torch.float32)
emb = model._extract_object_level_embeddings(x, bboxes, batch_idx)
print("emb_shape", tuple(emb.shape))
print("proj_params", sum(p.numel() for p in model.projection_head.parameters()))
