import torch


a = 1/9
b = 1/16
c = 1/256
d = -1/256

identity_kernel = torch.tensor([[.0, .0, .0],
                                [.0, 1.0, .0],
                                [.0, .0, .0]])

edge_detection_kernel = torch.tensor([[-1.0, 0.0, 1.0],
                                      [-1.0, 0.0, 1.0],
                                      [-1.0, 0.0, 1.0]])

edge_detection_RIDGE0 = torch.tensor([[.0, -1.0, .0],
                                     [-1.0, 4.0, -1.0],
                                     [.0, -1.0, .0]])
edge_detection_RIDGE1 = torch.tensor([[-1.0, -1.0, -1.0],
                                     [-1.0, 8.0, -1.0],
                                     [-1.0, -1.0, -1.0]])

sharpen_kernel = torch.tensor([[.0, -1.0, .0],
                                [-1.0, 5.0, -1.0],
                                [.0, -1.0, .0]])

box_blur = torch.tensor([[a, a, a],
                         [a, a, a],
                         [a, a, a]])

gaussian_blur3x3 = torch.tensor([[b*1, b*2, b*1],
                                [b*2, b*4, b*2],
                                [b*1, b*2, b*1]])

gaussian_blur5x5 = torch.tensor([[c*1, c*4, c*6, c*4, c*1],
                                 [c*4, c*16, c*24, c*16, c*4],
                                 [c*6, c*24, c*36, c*24, c*6], 
                                 [c*4, c*16, c*24, c*16, c*4],
                                 [c*1, c*4, c*6, c*4, c*1]])

unsharp_masking = torch.tensor([[d*1, d*4, d*6, d*4, d*1],
                                 [d*4, d*16, d*24, d*16, d*4],
                                 [d*6, d*24, d*-476, d*24, d*6], 
                                 [d*4, d*16, d*24, d*16, d*4],
                                 [d*1, d*4, d*6, d*4, d*1]])