# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Hoàng Quốc Việt
- MSSV: 2A202602563
- Email: 26ai.viethq@vinuni.edu.vn
- Link repo (fork): https://github.com/Catnip-harvest/K4-L2L3-DAY23-HoangQuocViet-2A202602563-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): commit cuối trên `main` (hash 40 ký tự nộp trên LMS); artifacts chấm điểm được sinh ở commit CP5 `e4ef9132f8e14025bee46e6b8676b41546d6a927`, sau lần sửa code E–H cuối cùng (CP4)

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`, `[0, 198]` (199 frame), `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `seed = 0`
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: precision 0.9701, recall 0.7004, tp 519 / fp 16 / fn 222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.1503 m, 502, 11.3437, 0, 239, 2.5226
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.1359 m, 502, 9.2668, 0, 239, 2.5226
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: xem bảng và phân tích ngay dưới.

| Mode | RMSE (m) | matches | ghost_track_frames | missed_gt_frames | precision_track | coverage = matches/det_tp |
|---|---|---|---|---|---|---|
| LiDAR | 0.1503 | 502 | 0 | 239 | 502/502 = 1.000 | 502/519 = 0.967 |
| LiDAR + camera | 0.1359 | 502 | 0 | 239 | 1.000 | 0.967 |

`rmse_fused − rmse_lidar = −0.0145 m` (fused tốt hơn, trong giới hạn ≤ 0.05 m).

**Đọc RMSE cùng số ghép, ghost và miss.**

- Hai mode có **cùng** `matches = 502`, `ghost_track_frames = 0`, `missed_gt_frames = 239`, `mean_confirmed_tracks = 2.5226`. Trong `grade_run.log`, ở **cả 199 frame**, trường `confirmed` và `matches` của record `lidar` và `fused` bằng nhau. Như vậy RMSE của hai mode được tính trên **cùng tập cặp track–xe**, nên so sánh RMSE là công bằng: camera không "mua" RMSE thấp bằng cách bỏ bớt cặp khó.
- Khác biệt duy nhất nằm ở sai số vị trí: `sum_sq_err` giảm từ 11.34 xuống 9.27 (−18%). Fused có sai số bình phương thấp hơn LiDAR ở 137/195 frame có cặp ghép. Theo từng track (`student/bonus/viz_summary.json`): T10 0.195 → 0.140 m, T1 0.122 → 0.092 m, T5 0.160 → 0.150 m, nhưng T0 xấu đi 0.148 → 0.166 m. Camera giúp trung bình, không giúp mọi track (lý do ở Bonus 3).
- `ghost = 0`: 16 detection FP của detector chỉ sinh track `initialized`/`tentative`, bị xoá trước khi đạt `score > 0.8`, nên không bao giờ thành confirmed (bằng chứng: Bonus 1).
- `missed_gt_frames = 239` = 222 lần detector bỏ sót (`det_fn`) + 17. Con số 17 = 22 GT-frame mất do **độ trễ xác nhận** (frame 0–3 lúc khởi động và 4 frame đầu của mỗi track mới, ví dụ frame 44–47, 60–63) trừ 5 GT-frame mà track confirmed đã **dự báo qua** một lần detector bỏ sót. Coverage 0.967 chỉ tính trên xe detector thấy (`det_tp = 519`), nên 222 lần bỏ sót của detector không trừ điểm tracking.

**Giới hạn của đo camera.** Đo camera là tâm hộp 2D ground-truth FRONT cộng nhiễu Gauss σ = 0.5 px có seed (`platform/fusion_lab/scripts/run_lab.py:97`), **không** phải output của detector ảnh. Kết quả fused vì vậy là cận trên lạc quan: một detector thật có nhiễu lớn hơn, có FP/FN và lệch tâm. Ngay cả nhãn GT cũng có sai lệch hệ thống so với tâm hộp 3D chiếu xuống ảnh: innovation camera trung bình γ_u = −11.9 px khi calibration đúng (Bonus 3). Lý do là tâm hộp 2D bao cả mặt hông xe, không trùng với hình chiếu tâm 3D. Kết quả này không đo hiệu quả một perception system độc lập với GT.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

Đã kiểm tra lại: `grade_run.log` có 398 record (199 frame × 2 mode), không trùng `(mode, frame)`, mọi record thoả hai bất biến, detection của hai mode giống hệt nhau từng frame, và `validate_metrics_records` khớp `metrics.json`.

## Giải thích ngắn (Parts E–H — tự viết)

1. **Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?**
   - **LiDAR:** `z = (x, y, z)` là tâm hộp 3D (m), `R = diag(0.1², 0.1², 0.1²)` m². `h(x)` là phép biến đổi cứng sang hệ LiDAR, tuyến tính, nên `H` là hằng số 3×6 (khối xoay, cột vận tốc bằng 0).
   - **Camera:** `z = (u, v)` là pixel, `R = diag(5², 5²)` px² (`student/workspace/camera_fusion.py:101`). `h(x)` là phép chiếu pinhole phi tuyến: `u = c_i − f_i·y_s/x_s`, `v = c_j − f_j·z_s/x_s` (`camera_fusion.py:82-83`), sau khi đổi `p_s = R p + t`. EKF phải tuyến tính hoá bằng Jacobian `H` 2×6 do platform tính tại `x` hiện tại. Camera chỉ đo **hướng** (2 chiều) chứ không đo độ sâu, nên nó chủ yếu sửa vị trí ngang của track. Cùng công thức `ekf_update` (`kalman.py:128-130`: `K = P Hᵀ S⁻¹`, `x + Kγ`, `(I − KH)P`) dùng cho cả hai, chỉ khác `z`, `R`, `h`, `H`.
2. **Vì sao cần gating Mahalanobis trước khi gán?** Khoảng cách Euclid bỏ qua độ bất định: 1 m có thể rất gần với track mới sinh (vận tốc σ = 50 m/s), nhưng rất xa với track đã hội tụ. `d² = γᵀ S⁻¹ γ` (`association.py:29-42`) chuẩn hoá innovation theo `S = HPHᵀ + R`, đồng thời gộp đơn vị m và px vào cùng một thang. Nếu đo đúng là của track, `d²` phân phối χ² với `dim_meas` bậc tự do. Cổng `chi2.ppf(0.995, dim)` (`association.py:56`; 12.84 cho LiDAR, 10.60 cho camera) loại các cặp quá khó xảy ra *trước* khi greedy chọn cặp rẻ nhất. Nhờ vậy một track không bị kéo về FP hay về xe bên cạnh. Với camera, `in_fov` được kiểm tra trước (`association.py:77`), nên track sau lưng camera không bao giờ bị chiếu.
3. **Track-then-fuse hay fuse-then-track? Chỉ ra trên log.** Đây là **track-then-fuse**. Chỉ có một tracker; mỗi frame predict một lần, rồi `associate_and_update` với LiDAR, rồi với camera (`run_lab.py:200, 202, 209`). Trên log: ở cả 199 frame, record `lidar` và `fused` có **cùng** `confirmed` và `matches`, `mean_confirmed_tracks` bằng nhau (2.5226). Ví dụ frame 100: cả hai mode `confirmed=3, matches=3`, nhưng `sum_sq_err` là 0.0601 (lidar) và 0.0488 (fused). Camera chỉ đổi **trạng thái** (sai số), không đổi tập track.
4. **Camera lệch calibration thì triệu chứng gì trên innovation/residual?** Innovation camera mất tính trung bình 0 và bị **lệch có hệ thống**, gần đúng `f_i·tan(δ)` px theo hướng lệch. NIS tăng lên trên mức kỳ vọng 2, và nhiều cặp bị cổng χ² từ chối. Đo trên segment này (Bonus 3): lệch yaw +0.25° → γ_u trung bình −11.9 → −20.0 px, NIS 2.25 → 4.55; +0.5° → −26.9 px, NIS 7.28, 38% track nhìn thấy bị từ chối; ≥ 1° → 97% bị từ chối. Nguy hiểm nhất là lệch **nhỏ**: nó lọt qua cổng nhưng kéo lệch trạng thái (RMSE fused 0.211 m ở +0.5°). Lệch lớn thì bị gating chặn gần hết, camera coi như tắt, RMSE quay về mức LiDAR (0.150 m ở 5°).
5. **Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng? Vì sao lidar quyết định score/init/delete còn camera chỉ EKF update?**
   - **Frame rỗng:** khi `meas_list` rỗng, không thể suy ra lượt này là LiDAR hay camera từ dữ liệu. Nhưng `manage_tracks` phải biết: lượt LiDAR rỗng nghĩa là mọi track trong FOV LiDAR đều **miss** (trừ score, có thể xoá). Lượt camera rỗng không được làm gì. Vì thế hàm luôn gọi `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)` (`association.py:143`), kể cả khi không có đo. Test `test_empty_lidar_frame_scores_then_deletes_exhausted_track` kiểm tra đúng đường này.
   - **Phân vai sensor:** LiDAR đo đủ 3D và có detector sinh ra track. Camera ở đây chỉ phủ FOV FRONT hẹp hơn, chỉ đo hướng, không có độ sâu nên không khởi tạo được trạng thái 3D. Nếu camera miss cũng trừ score, mọi xe ngoài FOV camera hoặc bị che sẽ bị xoá oan. Platform thực thi điều này: `handle_updated_track` chỉ ghi hit khi `sensor.name == "lidar"`, còn `manage_tracks` return ngay với camera (`platform/fusion_lab/tracking/manager.py:95, 111`). Bằng chứng số: `mean_confirmed_tracks` của hai mode bằng nhau.
6. **Điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xoá track.** (`student/workspace/track_management.py:44-90`)
   - **Khởi tạo** từ một đo LiDAR chưa ghép: score = 1/6, state `initialized`, P vị trí = `R_rot · R · R_rotᵀ`, P vận tốc = diag(50², 50², 5²).
   - **Hit LiDAR:** score += 1/6, tối đa 1; track chưa confirmed thì thành `tentative`. **Miss trong FOV LiDAR:** score −= 1/6.
   - **Xác nhận** khi `score > 0.8` (cần 5 hit liên tiếp từ lúc sinh: 5/6 ≈ 0.833).
   - **Giữ confirmed:** một miss chỉ đưa score 1 → 0.833 và state vẫn `confirmed`; không bao giờ hạ cấp.
   - **Xoá** (các điều kiện OR): `P[0,0] > 9` hoặc `P[1,1] > 9` m², tức σ vị trí > 3 m, bất kể score; confirmed với `score < 0.6` (từ 1.0 cần 3 miss liên tiếp); chưa confirmed với `score ≤ 0` (track mới sinh chỉ cần 1 miss).
   - Camera không gọi các hàm này.

## Bonus (không bắt buộc)

Toàn bộ bằng chứng sinh bởi `student/bonus/bonus_analysis.py`. Script không sửa `platform/` hay `student/artifacts/`. Nó cache detection của detector có sẵn rồi **phát lại đúng vòng tracking của `fusion-run-lab`**. Lệnh `check` xác nhận bản phát lại (không nhiễu calibration) cho kết quả **trùng khớp từng giá trị** với `metrics_lidar.json` và `metrics_fused.json` của lần chạy chấm điểm, nên các số bonus cùng nguồn với artifacts.

```bash
python student/bonus/bonus_analysis.py cache --config student/config/paths.yaml
python student/bonus/bonus_analysis.py all      # check + calibration + visualise + cvat
```

**Bonus 1 — Export track sang CVAT (`export_tracks_json`).**

- File:
  - `student/bonus/cvat_tracks.json`: xuất qua hook `fusion_lab.export_cvat.export_tracks_json`, gồm mọi track mỗi frame kèm `state`/`score`.
  - `student/bonus/cvat_tracks_bev.xml`: định dạng *CVAT for video 1.1*, 16 track, 566 box trên ảnh BEV 608×608, có thuộc tính mutable `state` (initialized/tentative/confirmed).
  - `cvat_events.json`, `cvat_summary.json`.
- Kết quả: lần chạy fused có **0 ghost confirmed và 0 lần đổi ID**. Mỗi xe GT được giữ bởi đúng một track ID suốt lúc nhìn thấy, khớp `ghost_track_frames = 0`. Thay vào đó, file chỉ ra **25 GT-frame "ghost bị chặn"**: 9 track (T2, T3, T4, T6, T11–T15) sinh từ detection FP, cách xe GT gần nhất > 2 m, và bị xoá khi còn `initialized`/`tentative`. Ví dụ trong `student/bonus/cvat_suppressed_ghost_frame010.png`: T2 (`initialized`, score 0.17) cách GT gần nhất 4.3 m. Đây là tác dụng của luật xác nhận.
- **Giới hạn trung thực:** máy làm bài không có Docker để chạy CVAT local, và tôi không tạo tài khoản CVAT online. Vì vậy **chưa có ảnh chụp import trong CVAT**, chỉ có file đúng định dạng import và ảnh BEV tự render. Ảnh BEV cho task CVAT được sinh ở `data/bonus_cache/bev_frames/` (không commit vì là dữ liệu trích từ dataset).

**Bonus 2 — Trực quan hoá track và đo.**

- `student/bonus/viz_bev_camera_update.png`: BEV frame 134. Panel phải phóng to track T10: sau update LiDAR, track lệch GT 0.24 m theo XY; update camera dời nó 0.21 m về sát GT, còn **0.03 m**.
- `student/bonus/viz_front_camera_update.jpg`: ảnh FRONT cùng frame, gồm đo camera (dấu +), hình chiếu track sau update LiDAR (vòng xanh) và sau update camera (vòng đỏ), kèm inset phóng to T10.
- `student/bonus/viz_track_error_over_time.png`: sai số 3D theo frame của T10 (RMSE 0.195 → 0.140 m nhờ camera) và T0 (0.148 → 0.166 m, xấu đi).
- Nhận xét: camera hiệu quả nhất với track ở xa hoặc ở cạnh, nơi tâm hộp LiDAR dao động theo phía. Với T0 ở bên trái ảnh, độ lệch hệ thống của tâm hộp 2D (Bonus 3) kéo track lệch nhẹ.

**Bonus 3 — Phân tích calibration** (`student/bonus/calibration_study.json`, `student/bonus/calibration_drift.png`).

Làm lệch extrinsic camera mà tracker tin (ảnh thật giữ nguyên): xoay quanh trục "up" của camera một góc δ, và một mức dịch 0.5 m sang trái. Cổng χ²(2, 0.995) = 10.60. "Từ chối" = track confirmed nằm trong FOV camera mà `d²` nhỏ nhất tới mọi đo camera vẫn ≥ cổng.

| Lệch | f·tanδ (px) | Số update camera | γ_u TB (px) | \|γ_u\| TB (px) | NIS TB | Tỉ lệ bị gate từ chối | RMSE fused (m) | − RMSE LiDAR (m) |
|---|---|---|---|---|---|---|---|---|
| yaw −0.5° | −18.2 | 504 | +5.9 | 11.2 | 2.37 | 2.0% | 0.1936 | +0.043 |
| yaw −0.25° | −9.1 | 508 | −2.9 | 8.2 | 1.43 | 1.4% | 0.1464 | −0.004 |
| **0 (gốc)** | 0 | 509 | −11.9 | 12.9 | 2.25 | 1.2% | **0.1359** | **−0.015** |
| yaw +0.25° | 9.1 | 486 | −20.0 | 20.9 | 4.55 | 6.1% | 0.1732 | +0.023 |
| yaw +0.5° | 18.2 | 322 | −26.9 | 27.7 | 7.28 | 37.9% | 0.2112 | **+0.061** |
| yaw +1° | 36.4 | 20 | −75.0 | 83.1 | 3.51 | 97.4% | 0.1726 | +0.022 |
| yaw +2° | 72.7 | 19 | −80.0 | 86.3 | 3.92 | 97.6% | 0.1760 | +0.026 |
| yaw +5° | 182.2 | 28 | −51.9 | 53.6 | 2.04 | 97.0% | 0.1503 | 0.000 |
| dịch trái 0.5 m | — | 24 | −68.0 | 75.8 | 3.99 | 96.6% | 0.1621 | +0.012 |

Ở mọi mức lệch, `matches = 502`, `ghost = 0`, `missed = 239` không đổi, đúng với track-then-fuse: lệch camera chỉ làm hỏng trạng thái, không làm mất hay sinh track.

Nhận xét:

- **Triệu chứng trên innovation:** γ_u trung bình dịch đơn điệu theo δ, gần với `f·tanδ` (+0.25° → thêm −8.1 px, dự đoán 9.1 px). |γ_u| và NIS tăng vượt mức kỳ vọng 2 của χ²(2). Residual không còn trung bình 0: đây là dấu hiệu nhận biết calibration drift.
- **Khi nào gating chặn:** lệch ≤ 0.25° (≤ ~9 px) nhỏ hơn độ rộng cổng: `S` gồm R = 25 px² cộng phần `HPHᵀ` (track hội tụ, σ vị trí cỡ 0.1 m ⇒ vài px). `d²` vẫn < 10.6, nên **gating không chặn** và đo lệch kéo track (+0.023 m). Ở 0.5°, 38% bị chặn, phần còn lại lọt qua với NIS cao: đây là **vùng nguy hiểm nhất**, RMSE xấu nhất (0.211 m, vi phạm ngưỡng nhất quán 0.05 m). Từ ≥ 1° (≥ 36 px ≈ 7σ), gần như mọi cặp đúng bị chặn. ~20 update còn lọt có |γ_u| ~ 80 px, khả năng cao là **gán nhầm** sang đo của xe bên cạnh rơi vào cổng. Camera coi như tắt, RMSE tiến về mức LiDAR (0.1503 m ở 5°, bằng LiDAR-only). Gating bảo vệ track khỏi lệch lớn, nhưng không phát hiện được lệch nhỏ.
- **Độ lệch −11.9 px ở calibration gốc không phải do extrinsic:** xoay −0.25° triệt gần hết độ lệch (−2.9 px, NIS 1.43) nhưng RMSE lại *xấu hơn* (0.1464 m). Nguyên nhân là mô hình đo: tâm hộp 2D của xe nhìn chéo bao cả mặt hông, nên không trùng hình chiếu tâm 3D. Sửa extrinsic để "bù" lỗi mô hình đo làm hỏng hình học thật. Đây cũng là lý do T0 (bên trái ảnh) xấu đi khi fused.

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Claude Code (mô hình Claude Opus 5.5, Anthropic)
- Dùng cho phần nào (hàm, câu hỏi, debug): viết code Part E–H (`kalman.py`, `camera_fusion.py`, `association.py`, `track_management.py`) theo các gợi ý `# vi: TODO` và test; viết script bonus `student/bonus/bonus_analysis.py`; cài môi trường, chạy `fusion-run-lab` và các lần chạy bonus; soạn nháp báo cáo này từ số liệu trong `metrics.json`, `grade_run.log` và các file JSON bonus
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức):
  - `pytest student/tests -q`: 128 passed, 0 failed, 0 xfailed.
  - Đối chiếu từng công thức với hướng dẫn kỹ thuật và tham số `tracking_params.py`, ví dụ test số học `test_predict_numeric_state_and_covariance` và Mahalanobis = 2.
  - Chạy `fusion-run-lab --fusion compare --seed 0` trên frame 0–198, rồi tự kiểm tra lại `grade_run.log`: 398 record, bất biến, `validate_metrics_records`.
  - Bản phát lại trong script bonus tái tạo đúng `metrics_lidar.json`/`metrics_fused.json` trước khi dùng số bonus.
  - Mọi số trong báo cáo được lấy từ các file đó, không nhập tay.
  - `python tools/check_submission.py` báo sẵn sàng nộp.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa) — không sửa
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
