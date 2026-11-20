# HƯỚNG DẪN UPLOAD TRỰC TIẾP LÊN GITHUB (KHÔNG DÙNG GIT COMMAND)

1. Tải `AAVAIL_Capstone_GitHub_Upload.zip` rồi **giải nén** ra một thư mục trên máy. Không upload mỗi file ZIP lên GitHub; người chấm cần thấy từng file mã nguồn và cấu trúc folder.
2. Truy cập **https://github.com/new**, đăng nhập GitHub.
3. **Repository name:** `aavail-customer-churn-capstone`; **Visibility:** `Public`. Không chọn tạo README mặc định vì bộ này đã có `README.md`.
4. Nhấn **Create repository**. Trên màn hình repo trống, nhấn **uploading an existing file**. Nếu repo đã có nội dung thì chọn **Add file → Upload files**.
5. Mở **thư mục đã giải nén** trên máy; chọn toàn bộ **nội dung bên trong** (folder `aavail`, `data`, `artifacts`, `scripts`, `tests`, `.github` và các file gốc như `README.md`, `Dockerfile`), kéo thả vào vùng upload của GitHub. Không kéo thả một folder mẹ khiến tất cả bị đặt sâu thêm một cấp.
6. Kiểm tra các đường dẫn xuất hiện: `aavail/api.py`, `tests/test_api.py`, `scripts/run_pipeline.py`, `artifacts/figures/05_model_vs_baseline.png`, `Dockerfile`, `.github/workflows/tests.yml`, `README.md`. Nếu không thấy `.github`, bật hiển thị file ẩn trên máy (ví dụ Ctrl+H trên Linux).
7. Nhập commit message: `Add complete AAVAIL churn capstone`, nhấn **Commit changes** (hoặc nút commit tương tự).
8. Mở tab **Actions** để xem kết quả kiểm thử tự động. Nếu GitHub yêu cầu bật workflow trong repo, hãy làm theo hướng dẫn trên màn hình.
9. Mở link repo bằng cửa sổ ẩn danh để xác nhận sinh viên khác truy cập được.
10. Nộp **URL trang repo** (không phải đường dẫn file ZIP) lên Coursera: `https://github.com/TEN_GITHUB_CUA_BAN/aavail-customer-churn-capstone`.

Nếu trình duyệt không cho thả cả folder, chia nhỏ lượt upload theo thư mục hoặc tạo trước đường dẫn bằng **Add file → Create new file**. Không cần `git init`, `git remote`, `git add` hoặc `git push`.

Theo tài liệu GitHub: upload bằng trình duyệt giới hạn 25 MiB/file và tối đa 100 files/lần. Bộ bài này nằm trong giới hạn trên.

Tài liệu chính thức:
- https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository
- https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
