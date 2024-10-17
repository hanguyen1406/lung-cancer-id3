from math import log2

# Hàm tính entropy
def tinh_entropy(nhan):
    tong_mau = len(nhan)
    nhan_1 = nhan.count('Có') / tong_mau
    nhan_0 = nhan.count('Không') / tong_mau
    if nhan_1 == 0 or nhan_0 == 0:
        return 0
    return -(nhan_1 * log2(nhan_1) + nhan_0 * log2(nhan_0))

# Hàm tính Gain thông tin
def gain_thong_tin(du_lieu, nhan, thuoc_tinh):
    # print('Dữ liệu:',du_lieu)
    
    print('Thuộc tính:',thuoc_tinh)
    tong_entropy = tinh_entropy(nhan)
    print(f'Entropy nhãn {nhan}: {tong_entropy}')
    gia_tri = list(set([mau[thuoc_tinh] for mau in du_lieu]))
    tong_mau = len(nhan)
    
    # Tính entropy có trọng số
    entropy_con = 0
    for gt in gia_tri:
        chi_tiet = [nhan[i] for i in range(len(nhan)) if du_lieu[i][thuoc_tinh] == gt]
        print(f'Chi tiết: {gt}',chi_tiet)
        entropy_con += (len(chi_tiet) / tong_mau) * tinh_entropy(chi_tiet)
    
    # Gain thông tin
    return tong_entropy - entropy_con

# Hàm xây dựng cây quyết định ID3
def xay_dung_cay(du_lieu, nhan, thuoc_tinh_con_lai, do_sau):
    print('\nVòng lặp tính gain mới\n')
    # Dừng nếu tất cả nhãn giống nhau
    if nhan.count(nhan[0]) == len(nhan):
        return nhan[0]
    
    # Dừng nếu không còn thuộc tính nào để phân chia
    if len(thuoc_tinh_con_lai) == 0:
        print(max(set(nhan), key=nhan.count))
        return max(set(nhan), key=nhan.count)
    
    # Tìm thuộc tính tốt nhất để phân chia
    gain_cao_nhat = -1
    thuoc_tinh_tot_nhat = None
    for tt in thuoc_tinh_con_lai:
        gain = gain_thong_tin(du_lieu, nhan, tt)
        if gain > gain_cao_nhat:
            gain_cao_nhat = gain
            thuoc_tinh_tot_nhat = tt
    
    # Tạo nhánh của cây quyết định
    cay = {thuoc_tinh_tot_nhat: {}}
    gia_tri_thuoc_tinh = list(set([mau[thuoc_tinh_tot_nhat] for mau in du_lieu]))
    
    for gt in gia_tri_thuoc_tinh:
        phan_du_lieu = [du_lieu[i] for i in range(len(du_lieu)) if du_lieu[i][thuoc_tinh_tot_nhat] == gt]
        phan_nhan = [nhan[i] for i in range(len(du_lieu)) if du_lieu[i][thuoc_tinh_tot_nhat] == gt]
        cay[thuoc_tinh_tot_nhat][gt] = xay_dung_cay(
            phan_du_lieu, 
            phan_nhan, 
            [tt for tt in thuoc_tinh_con_lai if tt != thuoc_tinh_tot_nhat], 
            do_sau + 1
        )
    
    return cay

# Hàm dự đoán
def du_doan(mau, cay):
    if not isinstance(cay, dict):
        return cay
    thuoc_tinh = list(cay.keys())[0]
    gia_tri = mau[thuoc_tinh]
    nhanh_con = cay[thuoc_tinh].get(gia_tri, None)
    if nhanh_con is None:
        return "Không rõ"
    return du_doan(mau, nhanh_con)

# Dữ liệu huấn luyện bằng tiếng Việt
du_lieu = [
    {'Thời tiết': 'Nắng', 'Nhiệt độ': 'Nóng', 'Độ ẩm': 'Cao', 'Gió': 'Không'},
    {'Thời tiết': 'Nắng', 'Nhiệt độ': 'Nóng', 'Độ ẩm': 'Cao', 'Gió': 'Có'},
    {'Thời tiết': 'Âm u', 'Nhiệt độ': 'Nóng', 'Độ ẩm': 'Cao', 'Gió': 'Không'},
    {'Thời tiết': 'Mưa', 'Nhiệt độ': 'Mát', 'Độ ẩm': 'Cao', 'Gió': 'Không'},
    {'Thời tiết': 'Mưa', 'Nhiệt độ': 'Lạnh', 'Độ ẩm': 'Bình thường', 'Gió': 'Không'},
    {'Thời tiết': 'Mưa', 'Nhiệt độ': 'Lạnh', 'Độ ẩm': 'Bình thường', 'Gió': 'Có'},
    {'Thời tiết': 'Âm u', 'Nhiệt độ': 'Lạnh', 'Độ ẩm': 'Bình thường', 'Gió': 'Có'},
    {'Thời tiết': 'Nắng', 'Nhiệt độ': 'Mát', 'Độ ẩm': 'Cao', 'Gió': 'Không'},
    {'Thời tiết': 'Nắng', 'Nhiệt độ': 'Lạnh', 'Độ ẩm': 'Bình thường', 'Gió': 'Không'},
    {'Thời tiết': 'Mưa', 'Nhiệt độ': 'Mát', 'Độ ẩm': 'Bình thường', 'Gió': 'Không'},
    {'Thời tiết': 'Nắng', 'Nhiệt độ': 'Mát', 'Độ ẩm': 'Bình thường', 'Gió': 'Có'},
    {'Thời tiết': 'Âm u', 'Nhiệt độ': 'Mát', 'Độ ẩm': 'Cao', 'Gió': 'Có'},
    {'Thời tiết': 'Âm u', 'Nhiệt độ': 'Nóng', 'Độ ẩm': 'Bình thường', 'Gió': 'Không'},
    {'Thời tiết': 'Mưa', 'Nhiệt độ': 'Mát', 'Độ ẩm': 'Cao', 'Gió': 'Có'},
]

# Nhãn kết quả (Có nghĩa là có thể chơi tennis, Không là không thể)
nhan = ['Không', 'Không', 'Có', 'Có', 'Có', 'Không', 'Có', 'Không', 'Có', 'Có', 'Có', 'Có', 'Có', 'Không']

# Huấn luyện mô hình
thuoc_tinh_con_lai = list(du_lieu[0].keys())
print(thuoc_tinh_con_lai)
cay_quyet_dinh = xay_dung_cay(du_lieu, nhan, thuoc_tinh_con_lai, 0)

# Kiểm tra kết quả dự đoán cho mẫu mới
mau_moi = {'Thời tiết': 'Nắng', 'Nhiệt độ': 'Mát', 'Độ ẩm': 'Cao', 'Gió': 'Không'}
du_doan_ket_qua = du_doan(mau_moi, cay_quyet_dinh)

print("Cây quyết định:", cay_quyet_dinh)
print("Dự đoán cho mẫu mới:", du_doan_ket_qua)
