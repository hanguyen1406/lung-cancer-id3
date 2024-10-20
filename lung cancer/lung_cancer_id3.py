from math import log2

# Hàm tính entropy
def tinh_entropy(nhan):
    tong_mau = len(nhan)
    nhan_1 = nhan.count(1) / tong_mau
    nhan_0 = nhan.count(0) / tong_mau
    if nhan_1 == 0 or nhan_0 == 0:
        return 0
    return -(nhan_1 * log2(nhan_1) + nhan_0 * log2(nhan_0))

# Hàm tính Gain thông tin
def gain_thong_tin(du_lieu, nhan, thuoc_tinh):
    tong_entropy = tinh_entropy(nhan)
    gia_tri = list(set([mau[thuoc_tinh] for mau in du_lieu]))
    tong_mau = len(nhan)
    
    # Tính entropy có trọng số
    entropy_con = 0
    for gt in gia_tri:
        chi_tiet = [nhan[i] for i in range(len(nhan)) if du_lieu[i][thuoc_tinh] == gt]
        entropy_con += (len(chi_tiet) / tong_mau) * tinh_entropy(chi_tiet)
    
    # Gain thông tin
    return tong_entropy - entropy_con

# Hàm xây dựng cây quyết định ID3
def xay_dung_cay(du_lieu, nhan, thuoc_tinh_con_lai, do_sau):
    # Dừng nếu tất cả nhãn giống nhau
    if nhan.count(nhan[0]) == len(nhan):
        return nhan[0]
    
    # Dừng nếu không còn thuộc tính nào để phân chia
    if len(thuoc_tinh_con_lai) == 0:
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
        phan_nhan = [nhan[i] for i in range(len(nhan)) if du_lieu[i][thuoc_tinh_tot_nhat] == gt]
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

# Đọc dữ liệu từ file
data = open('./lung_cancer_traning.csv', encoding='utf8').readlines()

# Thuộc tính cho các dữ liệu
attributes = ["tuổi", "giới tính", "ô nhiễm không khí", "sử dụng rượu", "dị ứng bụi", "nguy cơ nghề nghiệp", 
              "nguy cơ di truyền", "bệnh phổi mãn tính", "chế độ ăn uống cân đối", "béo phì", "hút thuốc", 
              "hút thuốc thụ động", "đau ngực", "ho ra máu", "mệt mỏi", "giảm cân", "khó thở", "thở khò khè", 
              "khó nuốt", "ngón tay dùi trống", "cảm lạnh thường xuyên", "ho khan", "ngáy", "ung thư phổi"]

# Tiền xử lý dữ liệu
data_traning = []
for row in data:
    # Loại bỏ ký tự xuống dòng và tách các giá trị theo dấu phẩy
    values = row.strip().split(',')
    
    # Tạo từ điển cho từng hàng dữ liệu
    entry = {attributes[i]: int(values[i]) for i in range(len(attributes))}
    
    # Thêm từ điển vào danh sách data_traning
    data_traning.append(entry)

def group_age(age):
    if age >= 0 and age <= 100:
        group = (age // 5) + 1  # Nhóm từ 1 đến 20
        return group
    return None  # Trường hợp tuổi ngoài khoảng 0-100

# Cập nhật giá trị tuổi trong danh sách data_traning
for entry in data_traning:
    entry['tuổi'] = group_age(entry['tuổi'])

labels = []

# Duyệt qua danh sách và trích xuất nhãn từ cột cuối cùng, thay đổi nhãn theo yêu cầu
for entry in data_traning:
    if entry['ung thư phổi'] == "Low":
        labels.append(0)
    else:
        labels.append(1)

# Loại bỏ thuộc tính 'ung thư phổi' vì đó là nhãn
attributes.pop()

# Huấn luyện mô hình
print(attributes)
cay_quyet_dinh = xay_dung_cay(data_traning, labels, attributes, 0)

# Hàm kiểm tra mô hình và tính toán độ chính xác
def kiem_tra_mo_hinh(test_data, test_labels, cay):
    dung = 0  # Số lượng dự đoán đúng
    sai = 0   # Số lượng dự đoán sai

    for i, mau in enumerate(test_data):
        du_doan_ket_qua = du_doan(mau, cay)
        thuc_te = test_labels[i]

        if du_doan_ket_qua == thuc_te:
            dung += 1
        else:
            sai += 1
    
    # Tính phần trăm độ chính xác
    do_chinh_xac = (dung / len(test_data)) * 100
    
    # Thống kê kết quả
    print(f"Số lượng dự đoán đúng: {dung}")
    print(f"Số lượng dự đoán sai: {sai}")
    print(f"Độ chính xác của mô hình: {do_chinh_xac:.2f}%")
    
    return do_chinh_xac

# Đọc file test và tiền xử lý tương tự như file huấn luyện
test_data = open('./lung_cancer_testing.csv', encoding='utf8').readlines()

# Tiền xử lý dữ liệu test
test_data_processed = []
test_labels = []
for row in test_data:
    values = row.strip().split(',')
    # Tạo từ điển cho từng hàng dữ liệu
    entry = {attributes[i]: int(values[i]) for i in range(len(attributes))}
    
    # Nhóm tuổi
    entry['tuổi'] = group_age(entry['tuổi'])
    
    # Thêm vào danh sách test_data_processed và lưu nhãn
    test_data_processed.append(entry)
    
    # Thay đổi nhãn tương ứng với yêu cầu
    if values[-1] == "Low":
        test_labels.append(0)
    else:
        test_labels.append(1)

# Kiểm tra mô hình
kiem_tra_mo_hinh(test_data_processed, test_labels, cay_quyet_dinh)
