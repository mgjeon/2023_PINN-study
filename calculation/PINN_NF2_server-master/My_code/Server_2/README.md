1. data_download
SHARP CEA 데이터 다운로드

2. extrapolation
Magnetic field 계산 + E/E_free 값 저장

3. csv_vtk
2에서 계산한 에너지 값에 대해서 csv파일 생성

추가로 특정 시점의 B, B_pot의 vtk파일 생성

4. Kusano
E, E_pot을 계산하여 csv파일 생성

5. series
csv파일을 이용하여 그래프 그리기

 # Anoter loss
boundary_b = b[:n_boundary_coords]
# mask = (torch.abs(boundary_b - b_true) - b_err) > 0
# # print(torch.abs(boundary_b - b_true))
# # print(b_err)
# # print(mask)
# b_diff2 = (torch.abs(boundary_b - b_true) - b_err)*mask
# print('b_diff2', b_diff2.shape)
# print(b_diff2.sum())
# b_diff2 = torch.mean(b_diff2.pow(2).sum(-1))
# print(iter)
# print((b_err).shape)
# print('b_err > 0', (b_err > 0).sum())
# print('b_err < 0', (b_err < 0).sum())
# print(b_err[b_err < 0])
# print('b_diff2', b_diff2)
b_diff = torch.clip(torch.abs(boundary_b - b_true) - b_err, 0)
# print('b_diff', b_diff.shape)
# print(b_diff.sum())
b_diff = torch.mean(b_diff.pow(2).sum(-1))
# print('b_diff', b_diff, b_diff.shape)
# print('b_err', b_err.sum())
# b_diff3 = torch.clip(torch.abs(torch.abs(boundary_b) - torch.abs(b_true)) - torch.abs(b_err), 0)
# b_diff3 = torch.mean(b_diff3.pow(2).sum(-1))
# print('b_diff3', b_diff3)

