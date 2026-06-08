# SR-GNN: Regularization-by-Decoupling cho Dự đoán Liên kết Thời gian theo hướng Inductive, kèm một Bộ đọc Vòng đời Trung thực và Có thể Can thiệp

*Bản thảo ED02 — paper toàn hệ thống. Mục tiêu: NeurIPS / ICML / ICLR. Mọi con số được báo cáo đều truy vết được tới một file JSON kết quả hoặc một job ID liệt kê trong phụ lục; những phát biểu chưa được kiểm chứng đều được giới hạn phạm vi rõ ràng. Mọi đại lượng `±` đều là độ lệch chuẩn mẫu (sample standard deviation, n−1) trên các seed được báo cáo.*

---

## Abstract (Tóm tắt)

Dự đoán liên kết thời gian liên tục (continuous-time temporal link prediction) đã hội tụ về hai họ kiến trúc: các mạng bộ nhớ (memory networks) mang một trạng thái học được theo từng node (JODIE [Kumar et al., 2019], TGN [Rossi et al., 2020], DyRep [Trivedi et al., 2019]) và các bộ mã hóa (encoder) theo cơ chế attention hoặc walk trên lân cận thời gian (TGAT [Xu et al., 2020], CAWN [Wang et al., 2021], GraphMixer [Cong et al., 2023]). Cả hai họ đều mạnh trong chế độ transductive nhưng suy giảm dưới giao thức *inductive* (quy nạp) khó hơn, nơi các tương tác kiểm thử liên quan tới các node chưa từng thấy ở thời điểm huấn luyện, và không họ nào cung cấp một cơ chế để trả lời các câu hỏi phản thực (counterfactual) về tương lai của một cạnh ("điều gì xảy ra nếu cặp này bị buộc phải chết?").

Chúng tôi trình bày SR-GNN, một mô hình hai luồng (two-stream) mà lựa chọn thiết kế trung tâm là một thao tác **detach** (tách rời): một backbone liên tục (bộ mã hóa sự kiện → bộ ước lượng trạng thái cạnh theo từng cặp đa tín hiệu → bộ nhớ coupled-GRU) chỉ được định hình bởi một mục tiêu tiết kiệm (parsimony) và được giữ *hoàn toàn stop-gradient* khỏi đầu dự đoán liên kết (link-prediction head), vốn đọc biểu diễn đã đóng băng của backbone thông qua một bộ giải mã vòng đời (lifecycle decoder) ký hiệu (symbolic). Chúng tôi cho thấy **regularization-by-decoupling** (chính quy hóa bằng cách tách rời) này không phải là một lỗi mà là nguồn gốc lợi thế inductive của SR-GNN: trên benchmark CoEdit, mô hình đầy đủ đạt **0.9885 ± 0.0035** average precision (AP) inductive, **+13.5 điểm** so với baseline tốt nhất trong sáu baseline dưới một giao thức đồng nhất đã được kiểm toán rò rỉ (leak-audited); việc ghép cùng đầu đó theo kiểu end-to-end *làm sụp đổ* AP inductive xuống **0.7672 ± 0.0107** (ba seed), một khoảng cách tách rời **+22.1 ± 1.4 điểm** với mọi seed đều trên 20 điểm. Trên Wikipedia và MOOC, mô hình cân bằng tốt nhất qua ba seed: TGAT nhỉnh hơn nó về inductive trên Wikipedia (0.998 so với 0.996) nhưng lại sụp đổ về transductive ở đó (0.658), trong khi SR-GNN mạnh ở cả hai giao thức.

Trên nền tảng đó chúng tôi xây dựng một **bộ đọc vòng đời phân cấp (hierarchical lifecycle readout)** phơi bày một mô hình nhân quả cấu trúc (structural-causal model, SCM) năm trạng thái trung thực (faithful) và có thể can thiệp (intervene-able) cho mỗi cạnh *mà không chạm vào đường AP* (được chứng minh bằng tính bất biến điểm số chính-xác-bằng-không). SCM hỗ trợ phản thực, đã kiểm chứng trên ba seed trên CoEdit: buộc một cạnh vào DEATH làm rớt xác suất tồn tại được dự đoán của nó ở ≥99% các cặp trên mọi seed, một can thiệp tăng tổng hợp lật DECAY→REINFORCE cho 0.9999 ± 0.0002, và mọi can thiệp đều có thể đảo ngược chính xác. Cuối cùng, chúng tôi báo cáo một tín hiệu độ tin cậy (confidence) dựa trên tính nhất quán nhân quả (causal-coherence) và trung thực về phạm vi của nó: đây là một thước đo *tự-nhất-quán (self-consistency)* ổn định (0.9985 ± 0.0015 AUC), *không phải* một bộ dự báo lỗi ngoại tại (external error) đã được kiểm chứng — một tuyên bố mà chúng tôi rút lại (retract) một cách tường minh sau khi một kết quả đơn-seed đã thất bại trong việc tái lập.

---

## 1. Giới thiệu

Các đồ thị tương tác thời gian — người dùng chỉnh sửa trang, sinh viên truy cập các module khóa học, các tài khoản giao dịch — là những chuỗi cạnh có dấu thời gian (timestamped edges). Một cách hình thức, một luồng (stream) là một tập có thứ tự các sự kiện $\mathcal{E} = \{(u_i, v_i, t_i, x_i)\}_{i=1}^{N}$ với $t_1 \le t_2 \le \dots \le t_N$, trong đó $x_i$ là một vector đặc trưng cạnh tùy chọn. Tác vụ kinh điển là *dự đoán liên kết tương lai (future link prediction)*: cho lịch sử $\mathcal{H}_t = \{(u_i, v_i, t_i, x_i) : t_i < t\}$, hãy chấm điểm xem một cạnh ứng viên $(u, v)$ có xảy ra tại thời điểm $t$ hay không. Mô hình xuất ra một xác suất $\hat{p}(u, v, t) = \sigma(f_\theta(u, v, t, \mathcal{H}_t))$ và được huấn luyện bằng binary cross-entropy so với các positive quan sát được và các negative được lấy mẫu.

Có hai giao thức đánh giá quan trọng, và chúng tưởng thưởng cho những thiên kiến quy nạp (inductive bias) rất khác nhau. Các cạnh kiểm thử **transductive** tái sử dụng các node huấn luyện, nên một mô hình có thể dựa vào một embedding theo từng node đã được ghi nhớ trong quá trình huấn luyện. Các cạnh kiểm thử **inductive** liên quan tới ít nhất một node chưa từng thấy trong huấn luyện, nên bất kỳ embedding theo từng node nào cũng chưa được khởi tạo hoặc ngẫu nhiên tại thời điểm kiểm thử. Inductive là chế độ liên quan tới triển khai thực tế — người dùng, trang và tài khoản mới liên tục xuất hiện — và là chế độ khó hơn, bởi mô hình phải chấm điểm cạnh từ các động lực học *có thể chuyển giao (transferable)* thay vì từ danh tính node. Một mô hình chấm điểm tốt theo transductive nhưng sụp đổ theo inductive thì đã ghi nhớ tổng thể huấn luyện chứ không học được quá trình tương tác.

Các phương pháp tân tiến nhất phân tách theo một trục biểu diễn: các mô hình *memory* (JODIE, TGN, DyRep) duy trì một trạng thái hồi quy (recurrent) theo từng node được cập nhật tại mỗi sự kiện, trong khi các mô hình *encoder* (TGAT, CAWN, GraphMixer) tính biểu diễn tức thời (on the fly) từ lân cận thời gian (§2 trình bày chi tiết từng phương pháp). Trên cả hai họ, biểu diễn được huấn luyện *end-to-end* so với mục tiêu dự đoán liên kết — backbone là bất kỳ thứ gì tối thiểu hóa loss xếp hạng (ranking loss) — và chúng tôi quan sát hai hệ quả.

1. **Ghép end-to-end mời gọi overfitting theo danh tính.** Khi link loss có quyền ghi vào backbone, gradient descent được tự do mã hóa bất kỳ đặc trưng nào làm giảm sai số xếp hạng huấn luyện, kể cả các đặc trưng chỉ đơn thuần *nhận dạng (identify)* các node huấn luyện — được tưởng thưởng theo transductive nhưng là khối lượng chết (dead weight) theo inductive, vì node kiểm thử là mới, nên mô hình phải trả một khoản "thuế inductive".
2. **Trạng thái học được vừa mờ đục vừa trơ.** Không có chỗ nào trong một vector bộ nhớ node để hỏi "cặp này đang được củng cố hay đang suy tàn?", và cũng không có toán tử nào để *can thiệp (intervene)* lên trạng thái đó và đọc ra tác động. Các quá trình thời gian có một vòng đời (lifecycle) tự nhiên — sinh ra, được củng cố, suy tàn, chết — nhưng các kiến trúc tiêu chuẩn không phơi bày gì trong số đó.

SR-GNN có một lập trường khác trên cả hai điểm. Chúng tôi giữ một backbone liên tục phong phú — một bộ mã hóa sự kiện, một toán tử trạng thái cạnh theo từng cặp đa tín hiệu được neo vào cường độ (intensity) của quá trình Hawkes [Hawkes, 1971], và một bộ nhớ node coupled-GRU — nhưng chúng tôi **tách rời nó khỏi link-prediction loss bằng một stop-gradient tường minh**. Backbone chỉ được định hình bởi một mục tiêu tiết kiệm biến phân (variational parsimony objective) (một bộ chính quy KL theo tinh thần của information bottleneck [Tishby et al., 2000] và VAE [Kingma & Welling, 2014]) cộng với các định luật cập nhật thống kê theo từng cặp mang tính tất định. Đầu dự đoán liên kết đọc backbone *đã đóng băng* này qua một bộ giải mã vòng đời ký hiệu nhỏ. Stop-gradient về mặt khái niệm chính là cùng một thiết bị ổn định việc học siamese tự giám sát (self-supervised siamese learning) [Chen & He, 2021]: nó ngăn một nhánh khỏi việc làm sụp đổ biểu diễn về một lối tắt (shortcut).

Các đóng góp của chúng tôi là:

1. **Regularization-by-decoupling (kết quả lõi).** Tách rời backbone khỏi dự đoán liên kết là thành phần *duy nhất* nâng AP inductive trên CoEdit từ **0.7672 ± 0.0107** lên **0.9885 ± 0.0035** trong một A/B khớp seed qua ba seed (end-to-end so với decoupled, cùng code, cùng đầu; **+22.1 ± 1.4 điểm**, mọi seed đều trên 20 điểm và các khoảng theo từng seed không chồng lấn). Chúng tôi đưa ra bằng chứng A/B và một lý giải cơ chế: đầu không thể tái định hình backbone về danh tính node huấn luyện, nên backbone giữ một biểu diễn động lực học theo từng cặp tổng quát có thể chuyển giao sang các node chưa thấy. Xuyên suốt, **mô hình đầy đủ là config B**; một ablation *không-vòng-đời, không-tinh-chỉnh (no-lifecycle, no-tune)* được lược bớt (chỉ detach) đã đạt 0.928 — trên mức 0.853 của baseline tốt nhất — cho thấy chính sự tách rời, chứ không phải bộ máy vòng đời, gánh phần chiến thắng inductive (§6.2).
2. **Một toán tử trạng thái cạnh theo từng cặp đa tín hiệu** (cường độ Hawkes, thống kê khoảng cách Welford, các EWMA tái diễn (recurrence) và tốc độ (rate)) với một bộ ước lượng theo batch kiểu *đọc-trước-khi-ghi (read-before-write)* (`causal_batch`) sửa một lỗi cũ kỹ (staleness) im lặng trong nội bộ batch; bản sửa này có lợi cho AP (**+5.7 ± 0.2 pp** inductive trên CoEdit qua ba seed, trong config B đầy đủ).
3. **Một bộ giải mã vòng đời năm trạng thái phân cấp** (BIRTH / REINFORCE / DECAY / DEATH / IDLE) làm cho trạng thái DECAY trung gian có thể đạt được bằng argmax — một softmax phẳng trên một trục có thứ tự về mặt cấu trúc là không thể — *và có thể chứng minh là bất biến với đường AP (AP-path-invariant)* (thay đổi điểm số chính-xác-bằng-không).
4. **Một động cơ phản thực / can thiệp (counterfactual / intervention engine)**, đã kiểm chứng trên ba seed trên CoEdit: vòng đời được giải mã là một SCM hoạt động được. `do(state = DEATH)` đưa xác suất tồn tại được dự đoán về 0 ở ≥99% các cặp trên mọi seed; toàn bộ thang trạng thái là đơn điệu (monotone); một can thiệp tăng tổng hợp lật DECAY→REINFORCE cho 0.9999 ± 0.0002 qua các seed; các can thiệp có thể đảo ngược chính xác.
5. **Một nghiên cứu về độ tin cậy trung thực.** Chúng tôi xây dựng một điểm tính-nhất-quán-nhân-quả theo chuỗi-bước (walked-chain causal-coherence) và báo cáo — với bằng chứng ba seed — rằng đó là một thước đo tự-nhất-quán nội tại ổn định nhưng *không phải* một bộ dự báo lỗi ngoại tại đáng tin cậy, rút lại một tuyên bố đơn-seed trước đó.

Xuyên suốt, các con số AP và các tuyên bố kiến trúc đều được đối chiếu chéo với code và với một cuộc kiểm toán liêm chính (integrity audit) độc lập (§6.3).

---

## 2. Công trình liên quan (Related work)

**Temporal GNN dựa trên bộ nhớ.** JODIE [Kumar et al., 2019] ghép hai RNN cùng tiến hóa embedding của user và item với một toán tử phép chiếu (projection) dự đoán quỹ đạo tương lai của một embedding giữa các sự kiện; nó được tinh chỉnh cho thiết lập gợi ý (recommendation) transductive. TGN [Rossi et al., 2020] hợp nhất các phương pháp như vậy thành một module bộ nhớ node (một RNN trên các message theo từng node) cộng một lớp embedding graph-attention. DyRep [Trivedi et al., 2019] đặt việc học biểu diễn như một quá trình điểm (point process) chú ý theo thời gian với động lực học topo và tương tác tách biệt. Cả ba đều duy trì một trạng thái hồi quy theo từng node được huấn luyện end-to-end so với mục tiêu liên kết. SR-GNN cũng mang bộ nhớ — một module bộ nhớ node coupled-GRU — nhưng bộ nhớ của nó được định hình bởi một mục tiêu tiết kiệm, chứ không phải trực tiếp bởi link loss, và đó chính xác là sự khác biệt mà ablation của chúng tôi cô lập.

**Các phương pháp encoder và dựa trên walk.** TGAT [Xu et al., 2020] áp dụng self-attention trên lân cận thời gian với một bộ mã hóa thời gian theo hàm Bochner, loại bỏ nhu cầu lưu một trạng thái theo từng node và cho hỗ trợ inductive bản địa (native). CAWN [Wang et al., 2021] ẩn danh hóa (anonymize) một tập các causal walk bắt rễ tại cặp truy vấn và mã hóa các mẫu danh-tính-tương-đối của chúng, nắm bắt cấu trúc thời gian ở cấp motif. GraphMixer [Cong et al., 2023] bỏ hẳn attention: một bộ mã hóa thời gian cố định cộng một MLP trộn token (token-mixing) trên các liên kết gần nhất là đủ cạnh tranh với các mô hình nặng hơn nhiều, một kết quả thúc đẩy việc dùng các baseline mạnh, đơn giản. Các phương pháp này mạnh theo transductive nhưng, như các lần chạy khớp giao thức của chúng tôi cho thấy (§6.1), hành vi *inductive* của chúng phụ thuộc dataset: TGAT gần như hoàn hảo theo inductive trên Wikipedia nhưng yếu hơn nhiều theo transductive ở đó, một sự kỳ quặc về đặc trưng (featural quirk) mà chúng tôi quy cho các đặc trưng node tình cờ nhận dạng được các node Wikipedia chưa thấy.

**Benchmark và đánh giá công bằng.** Kết quả đồ thị thời gian nhạy với giao thức đánh giá, đặc biệt là chiến lược lấy mẫu negative và split inductive. Temporal Graph Benchmark [Huang et al., 2023] chuẩn hóa các split và một chế độ negative khó hơn, và cho thấy nhiều bảng xếp hạng đã công bố bị sắp xếp lại dưới một giao thức công bằng. Chúng tôi tuân theo bài học này bằng cách chạy *mọi* mô hình qua một harness chung duy nhất với một pool negative duy nhất và một bộ đánh giá đã được kiểm toán rò rỉ (§6.1, §6.3), báo cáo cả hai giao thức để một mô hình thắng giao thức này nhưng sụp đổ giao thức kia được hiển thị thay vì bị che giấu.

**Quá trình điểm và mô hình hóa vòng đời.** Quá trình Hawkes [Hawkes, 1971] mô hình hóa cường độ sự kiện tự kích thích (self-exciting), trong đó mỗi sự kiện trong quá khứ tạm thời nâng tốc độ của các sự kiện tương lai; chúng là tiên nghiệm (prior) thời gian liên tục tự nhiên cho các luồng tương tác bùng nổ (bursty) và đã được dùng để dẫn dắt các mô hình thời gian neural [Mei & Eisner, 2017]. Chúng tôi dùng một cường độ Hawkes theo từng cặp $\lambda$ cùng với các mô-men trực tuyến Welford [Welford, 1962] của khoảng cách giữa các sự kiện (inter-event gap) làm nền tảng liên tục mà bộ giải mã ký hiệu đọc vào. Trừu tượng vòng đời BIRTH→REINFORCE→DECAY→DEATH trên các thống kê này là lớp ký hiệu; đó là một bản tóm tắt thô, có thể diễn giải về việc một cặp đang ở đâu trong quỹ đạo tự kích thích của nó.

**Các góc nhìn neuro-symbolic và nhân quả.** Bộ giải mã vòng đời của chúng tôi cộng với ma trận chuyển trạng thái theo luật-nhân-quả (causal-rule transition matrix) của nó là một mô hình nhân quả cấu trúc (SCM) nhỏ trên các trạng thái cạnh, và động cơ can thiệp hiện thực hóa các thao tác `do(·)` kiểu Pearl [Pearl, 2009] trên SCM đó. Khác với các bộ giải thích hậu kỳ (post-hoc explainers) như GNNExplainer [Ying et al., 2019], vốn khớp một surrogate riêng để biện minh cho một dự đoán cố định, trạng thái ký hiệu ở đây là *cùng* một đại lượng được giám sát và được đo đạc, nên lời giải thích là trung thực theo cấu trúc (faithful by construction) thay vì gần đúng. Tính trung thực (faithfulness) — đặc tính rằng lời giải thích phản ánh tính toán thực tế của mô hình — là mối quan tâm trung tâm của phê phán về diễn giải (interpretability critique) [Rudin, 2019; Jacovi & Goldberg, 2020], và thiết kế hai-đầu của chúng tôi (§3.4) trao điều đó cho chúng tôi một cách có thể chứng minh được.

**Decoupling và stop-gradient như những bộ chính quy.** Dừng dòng gradient theo một đường của một mô hình hai-nhánh là một bộ ổn định đã được biết tới. SimSiam [Chen & He, 2021] cho thấy một stop-gradient trên một nhánh siamese ngăn sự sụp đổ biểu diễn mà không cần negative, với sự bất đối xứng giữa một nhánh được cập nhật và một nhánh đóng băng là cơ chế tránh các nghiệm tầm thường. Tách rời một biểu diễn khỏi một loss hạ nguồn (downstream) cũng có thể được đọc qua lăng kính information-bottleneck [Tishby et al., 2000; Alemi et al., 2017]: khi mục tiêu duy nhất chạm tới backbone là một số hạng nén/tiết kiệm, nó bị đẩy về một mã đủ tối thiểu (minimal sufficient code) thay vì một lối tắt theo tác vụ. SR-GNN áp dụng điều này cho dự đoán liên kết thời gian, nơi lối tắt mà chúng tôi chặn là việc ghi nhớ danh tính node và mục tiêu bảo vệ là một KL kiểu VAE.

**Định vị.** SR-GNN không phải mô hình bộ nhớ đầu tiên, cũng không phải mô hình đầu tiên dùng các cường độ Hawkes, cũng không phải mô hình thời gian neuro-symbolic đầu tiên. Tuyên bố của nó là *sự kết hợp*: **tách rời (decoupling)** biểu diễn khỏi link loss là cái làm cho một biểu diễn động lực học theo từng cặp tổng quát hóa theo inductive, và bộ đọc ký hiệu được tách rời đồng thời là một SCM vòng đời trung thực, có thể can thiệp, mà không tốn gì cho bộ dự đoán.

---

## 3. Phương pháp

![Hình A1: Kiến trúc hai luồng SR-GNN. Một backbone liên tục (bộ mã hóa sự kiện → toán tử trạng thái cạnh theo từng cặp đa tín hiệu → bộ nhớ coupled-GRU) chỉ được định hình bởi một KL tiết kiệm; biểu diễn edge_h của nó được truyền qua một stop-gradient (bức tường detach) vào một bộ đọc vòng đời ký hiệu sinh ra logit tồn tại được chấm điểm. Không có gradient dự đoán liên kết nào vượt qua bức tường.](figs/A1_two_stream_detach.png)

*Hình A1. Kiến trúc hai luồng SR-GNN. Luồng A (backbone liên tục) chỉ được huấn luyện bởi KL tiết kiệm biến phân; biểu diễn `edge_h` của nó vượt qua một stop-gradient trước Luồng B ký hiệu (giải mã vòng đời → bộ giải mã tồn tại → logit được chấm điểm), nên link-prediction loss không bao giờ chạm tới backbone — phát biểu kiến trúc của regularization-by-decoupling (§3.7).*

### 3.1 Hai luồng, một detach

SR-GNN là một mô hình hai luồng. Gọi một sự kiện là một cạnh có dấu thời gian $(u, v, t)$ với vector đặc trưng tùy chọn $x$.

**Luồng A — backbone liên tục.** Một bộ mã hóa sự kiện (một mạng tín hiệu liên tục dạng residual, CSN) ánh xạ các đặc trưng sự kiện và độ cũ (staleness) của nguồn $\Delta t = t - t_{\text{last}}(u)$ thành một biểu diễn theo từng sự kiện $e_{uv}$. Một **toán tử trạng thái cạnh** đa tín hiệu (ECTGv3) duy trì, theo từng cặp có thứ tự $(u,v)$, một dải nhỏ các thống kê đang chạy:

- một **cường độ tự kích thích Hawkes** $\lambda_{uv}$ được cập nhật tại mỗi sự kiện bởi $\lambda \leftarrow 1 + (\lambda - 1)\,e^{-\beta \Delta t}$, nên mỗi tương tác tạm thời nâng tốc độ của cặp và tốc độ suy giảm giữa các sự kiện;
- **trung bình và phương sai trực tuyến Welford** của khoảng cách giữa các sự kiện, được cập nhật bởi đệ quy ổn định số học $\mu_k = \mu_{k-1} + (g_k - \mu_{k-1})/k$, $M_k = M_{k-1} + (g_k - \mu_{k-1})(g_k - \mu_k)$, với $\sigma_k^2 = M_k/k$ [Welford, 1962];
- một **EWMA tái diễn (recurrence)** đếm số lần đồng xuất hiện lặp lại, và các **EWMA tốc độ nhanh/chậm** $r^f, r^s$ mà tỷ số $r^f/r^s$ là một tín hiệu nhịp điệu tăng-so-với-giảm;
- một **đỉnh tốc độ rò rỉ (leaky rate-peak)** theo dõi tốc độ gần đây cực đại với suy giảm chậm.

Một module coupled-GRU (DRGC) cập nhật một bộ nhớ theo từng node $m_u, m_v$ từ $e_{uv}$ và phát ra một số hạng KL tiết kiệm $\mathrm{KL}(q(z \mid m) \,\|\, p(z))$. Tín hiệu huấn luyện *duy nhất* của backbone là KL này (một mục tiêu tiết kiệm biến phân, được trọng số bởi một scalar đơn $\lambda_{\text{kl}}$) cộng các định luật cập nhật thống kê tất định ở trên; **không có gradient dự đoán liên kết nào chạm tới nó.**

**Luồng B — bộ đọc vòng đời ký hiệu.** Từ biểu diễn cạnh (đã detach) `edge_h`, một `StateObserver` sinh ra một trạng thái *hiện tại* mềm $s_t \in \Delta^4$ trên năm trạng thái $\{\text{IDLE}, \text{BIRTH}, \text{REINFORCE}, \text{DECAY}, \text{DEATH}\}$; một `TransitionPredictor` sinh ra các logit trạng thái-kế-tiếp; một mặt nạ vòng đời — một ma trận chuyển theo luật-nhân-quả $C \in \{0,1\}^{5\times 5}$ mã hóa các chuyển dịch hợp lệ (ví dụ DEATH không thể đứng trước BIRTH) — cùng với một cổng `ever_alive` định hình phân phối trạng thái-kế-tiếp $s_{t+1}^{\text{pos}}$; một `ExistenceDecoder` ánh xạ $s_{t+1}^{\text{pos}}$ thành logit tồn tại cạnh được **chấm điểm**.

**Thao tác detach.** Đầu vào của mọi module Luồng-B là `edge_h.detach()`. Logit được chấm điểm là logit của bộ giải mã tồn tại, và mọi đường từ nó tới backbone đều vượt qua một stop-gradient. Chúng tôi đã kiểm chứng trên CPU rằng `pred_loss.backward()` cho ra gradient chính xác bằng không trên cả 56 tensor tham số backbone, trong mọi cấu hình. Một đầu dự đoán không-detach có tồn tại trong code nhưng chỉ được dùng cho *ablation* end-to-end (config C, §6.2).

### 3.2 Toán tử theo từng cặp và bản sửa đọc-trước-khi-ghi (`causal_batch`)

Toán tử trạng thái cạnh phải đọc các thống kê *trước-sự-kiện (pre-event)* của mỗi cặp: việc chấm điểm sự kiện $i$ chỉ được dùng trạng thái tích lũy từ các sự kiện $j < i$, nếu không việc đánh giá sẽ rò rỉ nhãn. Kho lưu trữ theo batch ban đầu chụp nhanh (snapshot) trạng thái một lần mỗi minibatch, nên các sự kiện cùng-cặp lặp lại *bên trong* một batch đều đọc cùng một hàng cũ và chỉ lần ghi cuối cùng được giữ lại. Ở batch size 500 trên CoEdit, số đếm Welford bị chặn gần 6 ngay cả với các cặp chỉnh sửa 200+ lần, làm under-fold cường độ Hawkes và ghim chặt đỉnh tốc độ — điều này âm thầm vô hiệu hóa toàn bộ phân biệt DECAY-so-với-REINFORCE, vì sự phân biệt đó được đọc ra từ tỷ số tốc độ mà không bao giờ được tích lũy.

Bản sửa `causal_batch` phát lại (replay) các kênh tất định theo từng-sự-kiện theo thứ tự stream, nên lần xuất hiện thứ $k$ trong batch của một cặp đọc post-state của lần thứ $(k{-}1)$, trong khi việc chấm điểm vẫn nghiêm ngặt trước-cập-nhật (không tái rò rỉ). Một kiểm tra tương đương trên CPU khớp một tham chiếu theo từng-sự-kiện (batch=1) tới max $|\Delta| = 0.000$ trên mọi kênh, xác nhận việc phát lại là chính xác. Bản sửa **có lợi cho AP**: trong config B đầy đủ, bật `causal_batch` nâng AP inductive trên CoEdit 0.9312 → 0.9885 (**+5.7 ± 0.2 pp**, ba seed) và transductive 0.9920 → 0.9985 (+0.65 pp) — một trường hợp hiếm khi một bản sửa tính đúng đắn đồng thời cũng là một chiến thắng về độ chính xác, bởi các thống kê đã-sụp-đổ trước đó chính xác là những thống kê mà bộ giải mã vòng đời cần. Hiệu ứng nhất quán cả trong thiết lập được lược (không-vòng-đời), nơi một A/B đơn-seed tăng 0.7462 → 0.7907 (job 5467100).

### 3.3 Giải mã vòng đời phân cấp

Trong một softmax năm-lớp phẳng trên trục có thứ tự BIRTH→REINFORCE→DECAY→DEATH, lớp giữa (DECAY) chia khối xác suất của nó cho hai lớp hàng xóm và *không bao giờ có thể thắng argmax* ngay cả khi xác suất của nó là trung thực. Cụ thể, nếu nhịp điệu thực sự đặt một cặp vào DECAY, một đầu phẳng trải tín hiệu đó qua REINFORCE và DEATH; trên config cuối cùng (giải mã phân cấp, `decol_hier_v2`, `causal_batch`, seed 42; dump faithfulness $N=12000$, tập con tái diễn $\text{true\_occ}\ge 2$, $n=9157$) chúng tôi đo được một Spearman $\rho \approx -0.59$ ($p < 10^{-300}$) giữa xác suất DECAY được giải mã (đã hiệu chỉnh) `p_decay_cal` và tín hiệu nhịp-điệu-tăng theo từng cặp `slope_rel` — đúng dấu (ít tăng hơn $\Rightarrow$ nhiều khối DECAY hơn) — nhưng một bộ đọc phẳng trên cùng tín hiệu đó vẫn không thể đưa DECAY lên thành lớp argmax. Không có tinh chỉnh ngưỡng nào sửa được điều này — đó là một đặc tính của *hình học* đầu ra, không phải của hiệu chỉnh (calibration) — nên *cấu trúc* đầu ra phải thay đổi.

Bộ giải mã phân cấp phân tích phân phối trạng thái-kế-tiếp thành một cây quyết định trên các cổng (gate) trước-cập-nhật theo từng cặp $p_{\text{birth}}, p_{\text{alive}}, p_{\text{rising}} \in [0,1]$:

$$
\begin{aligned}
P(\text{BIRTH}) &= p_{\text{birth}} \\
P(\text{REINFORCE}) &= (1 - p_{\text{birth}})\, p_{\text{alive}}\, p_{\text{rising}} \\
P(\text{DECAY}) &= (1 - p_{\text{birth}})\, p_{\text{alive}}\, (1 - p_{\text{rising}}) \\
P(\text{DEATH}) &= (1 - p_{\text{birth}})\, (1 - p_{\text{alive}})
\end{aligned}
$$

Bốn số hạng này tổng bằng một theo cấu trúc. Giờ đây DECAY chỉ cạnh tranh với REINFORCE *bên trong* nhánh alive (sự phân chia $p_{\text{rising}}$), nên argmax-DECAY có thể đạt được bất cứ khi nào $p_{\text{alive}} > p_{\text{birth}}$ và $p_{\text{rising}} < 1/2$.

![Hình A3: Cây giải mã vòng đời phân cấp.](figs/A3_hier_decode_tree.png)

*Hình A3. Cây giải mã vòng đời phân cấp (§3.3): ba cổng theo từng cặp `p_birth`/`p_alive`/`p_rising` phân tích năm trạng thái sao cho DECAY trở nên có thể đạt được bằng argmax — bản sửa cấu trúc mà một softmax phẳng không thể cung cấp.*
 Mỗi cổng là $\sigma(\text{analytic\_prior} + \text{phần dư khả học nhỏ})$, được khởi tạo bằng không sao cho một cổng mới bằng đúng tiên nghiệm phân tích của nó; một cross-entropy *chống-sụp-đổ (de-collapse)* huấn luyện các phần dư so với một mục tiêu mềm dẫn xuất từ các thống kê đang chạy. Một tinh chỉnh (`decol_hier_v2`) tái neo (re-anchor) các tiên nghiệm alive/rising lên tín hiệu số-đếm-tái-diễn không bị nhiễu và đặt các số hạng mean/staleness có thể bị nhiễu sau một mặt nạ "có-lịch-sử (has-history)", nên sự nhiễu đặc trưng không thể đẩy một cặp tái diễn đang hoạt động về DEATH.

![Hình 6: Quỹ đạo vòng đời được giải mã của một cặp CoEdit thực (3178→7437, 42 sự kiện, khoảng 18.69 phút). Độ dốc tốc độ chỉnh sửa và các cổng vòng đời bám theo nhịp BIRTH→REINFORCE↔DECAY; p_alive luôn trên 0.5 (không có DEATH).](experiments/LAB/v3_3/results/lifecycle_pair_3178_7437_en.png)

*Hình 6. Vòng đời theo từng cặp được giải mã cho một cặp CoEdit thực (3178→7437, 42 sự kiện, khoảng 18.69 phút, config B, đã hiệu chỉnh `s_t1_cal`). Độ dốc tốc độ chỉnh sửa (tăng→giảm) và các cổng phân cấp bám theo nhịp điệu của chính cặp đó: 1 BIRTH, 20 REINFORCE, 21 DECAY, 0 DEATH; p_alive luôn nằm trong [0.579, 0.801] > 0.5, với các lần lật REINFORCE↔DECAY tại các điểm đổi dấu độ dốc. Bộ đọc bám theo nhịp điệu thực của dữ liệu thay vì một tiên nghiệm cố định.*

### 3.4 Hai đầu trạng thái-kế-tiếp: AP so với diễn giải

Một đặc tính thiết kế then chốt: có **hai** phân phối trạng thái-kế-tiếp. `s_t1_pos` (softmax chuyển đã có mặt nạ, đã gating) là đầu vào *duy nhất* của bộ giải mã tồn tại và do đó là thứ duy nhất ảnh hưởng đến điểm AP. `s_t1_cal` (cây phân cấp ở trên, tùy chọn có causal-police) nuôi CE chống-sụp-đổ, phép đo phân phối vòng đời, dump faithfulness, và động cơ phản thực — nhưng **không bao giờ** nuôi bộ giải mã tồn tại.

Chúng tôi đã kiểm chứng trên CPU rằng việc chuyển phẳng↔phân cấp và bật↔tắt chính sách nhân quả (causal policy) thay đổi `s_t1_cal` tới mức 0.999 trong khi thay đổi cả điểm số positive và negative đúng **chính xác** 0.000e+00. Điều này mạnh hơn "đầu diễn giải nhỏ": đó là *tính bất biến điểm số (score invariance)*, một đặc tính của đồ thị tính toán (computation graph), nên không lựa chọn nào trong bộ đọc ký hiệu có thể dịch chuyển AP bất kỳ lượng nào. (Tính bất biến này là chính xác *khi trọng số cố định* — tức đánh giá một mô hình đã huấn luyện với bộ đọc được bật/tắt. Nó bảo đảm bộ đọc ký hiệu **không có** hiệu ứng có hệ thống lên AP, vì đầu ra của nó không bao giờ đưa vào loss của bộ giải mã tồn tại; còn khi bộ máy bộ đọc được bật trong lúc *huấn luyện* — ví dụ `hier_causal_policy`, §6.2 — thêm op làm dịch luồng RNG của optimizer, nên hai lần chạy AP-trung-tính trong phạm vi nhiễu seed chứ không phải giống-hệt-byte.) Do đó bộ máy diễn giải và phản thực vừa không thể bơm phồng AP, vừa khiến AP không thể là một tạo tác (artifact) của việc giải mã vòng đời — cho phép chúng tôi báo cáo một bộ đọc ký hiệu trung thực *và* một AP cạnh tranh mà không có sự đánh đổi diễn-giải-so-với-độ-chính-xác thường thấy [Rudin, 2019].

### 3.5 Chính sách nhân quả trên trạng thái có thể diễn giải

Trạng thái có thể diễn giải được công bố `s_t1_cal` được chính quy hóa bởi hai bước mềm, khả vi, được tái chuẩn hóa (renormalized):

1. một **cổng `ever_alive`**: lá DEATH được nhân tỷ lệ bởi bộ tích lũy đã-từng-sống của cặp, và khối-chết được giải phóng được định tuyến tới trạng thái IDLE trước-sinh trung thực, nên một cặp chưa bao giờ sống không thể bị báo cáo là đã chết;
2. một **mặt nạ khả-thừa-nhận kỳ vọng mềm (soft expected-admissibility mask)** từ ma trận luật-nhân-quả $C$: một kỳ vọng mềm đầy đủ $\mathbb{E}_{s_t}[C]$ trên phân phối trạng-thái-hiện-tại (không argmax), trộn với một floor nhỏ sao cho một chuyển dịch bị cấm bị triệt tiêu khoảng 20× thay vì bị giết cứng (hard-killed).

Chúng tôi dùng dạng mềm vì một mặt nạ argmax cứng dễ vỡ: khi $s_t$ gần đồng nhất, argmax dao động (jitter) giữa các trạng thái gần-hòa, trong khi kỳ vọng mềm suy giảm một cách nhẹ nhàng. Chính sách này là **AP-trung-tính (AP-neutral)** theo tính bất biến của §3.4: qua ba seed {1, 7, 42}, huấn luyện config B có so với không có chính sách làm dịch AP CoEdit inductive tối đa $|\Delta| = 1.5\mathrm{e}{-3}$ (trung bình $\approx -1.7\mathrm{e}{-4}$, lẫn-dấu), nằm gọn trong độ lệch chuẩn giữa-các-seed $\pm 3.5\mathrm{e}{-3}$ (job 5511229). Chính sách chỉ ghi `s_t1_cal` (nhánh đã detach mà bộ giải mã tồn tại không bao giờ đọc), nên không có hiệu ứng AP có hệ thống; phần dư là tính ngẫu nhiên huấn luyện (training stochasticity), không phải một sự khớp giống-hệt-byte.

![Hình A4: Dải khả-thừa-nhận FSM vòng đời năm trạng thái (C_BAND_5).](figs/A4_lifecycle_fsm_band5.png)

*Hình A4. FSM vòng đời năm trạng thái và dải khả-thừa-nhận của nó (C_BAND_5). $C \in \{0,1\}^{5\times5}$ chỉ cho phép các chuyển dịch kề nhau dọc trục IDLE–BIRTH–REINFORCE–DECAY–DEATH, theo cả hai chiều; mặt nạ mềm $\mathbb{E}_{s_t}[C]$ triệt tiêu các chuyển dịch bị cấm ~20× thay vì giết cứng chúng. Bảo đảm chết-trước-khi-sinh dùng một cổng `ever_alive` phi-Markov riêng (lưu ý §3.5).*

**Lưu ý trung thực (được chuyển vào Limitations).** `ever_alive` là một bộ tích lũy phi-Markov mà ma trận không-bộ-nhớ $C$ không thể biểu diễn, nên bảo đảm chết-trước-khi-sinh được thực thi bởi một cổng *riêng* thay vì bởi $C$ — một "mùi cấu trúc (structural smell)" đã biết, không phải bị che giấu.

### 3.6 Toán tử chuyển theo từng cặp

Phân phối trạng thái-kế-tiếp được sinh bởi một toán tử chuyển theo từng cặp $T_{uv}$ thích nghi ma trận luật-nhân-quả chung $C$ vào các thống kê của riêng cặp đó. Chúng tôi tham số hóa $T_{uv} = C \odot (W + g(\phi_{uv}))$, trong đó $W = UV^\top$ là một toán tử cơ sở khả học hạng-thấp (low-rank) và $g(\phi_{uv})$ là một cổng nhỏ theo từng cặp được tính từ bản tóm tắt đặc trưng của cặp $\phi_{uv}$ (tỷ số tốc độ, số đếm tái diễn, độ dốc, độ cũ). Cơ sở hạng-thấp nắm bắt các khuynh hướng chuyển ở cấp tổng thể; cổng $g$ chuyên biệt hóa chúng theo từng cặp mà không trao cho mỗi cặp một ma trận đầy đủ tự do. Vì $g$ là một hàm tất định của các thống kê (đã detach), toán tử có thể tái dựng ngoại tuyến (offline) từ một hàng đặc trưng đã lưu, và đó chính là cái làm cho động cơ can thiệp của §4 là chính xác.

### 3.7 Sự tách rời, một cách chính xác

Tổng loss là

$$
\mathcal{L} = \underbrace{\mathcal{L}_{\text{BCE}}(\hat{p}, y)}_{\text{prediction}} \;+\; \lambda_{\text{kl}}\,\mathrm{KL}\big(q(z\mid m)\,\|\,p(z)\big) \;+\; \lambda_{\text{dc}}\,\mathcal{L}_{\text{decol-CE}}(s_{t+1}^{\text{cal}}) ,
$$

và backward theo từng thành phần (đã chứng minh trên CPU, config B) cho thấy ba lộ trình gradient rời nhau:

1. **Backbone** (56 tensor) ← chỉ KL tiết kiệm và các định luật tất định. `pred_loss.backward()` cho gradient backbone $= 0$ trên cả 56 tensor.
2. **Đầu FSM / tồn tại** (đường `s_t1_pos`) ← BCE dự đoán liên kết $\mathcal{L}_{\text{BCE}}$.
3. **Các đầu phân cấp** (đường `s_t1_cal`) ← chỉ CE chống-sụp-đổ $\mathcal{L}_{\text{decol-CE}}$.

Stop-gradient trên `edge_h` chặn mọi gradient Luồng-B khỏi backbone. Đây là phát biểu hình thức của "regularization-by-decoupling": biểu diễn được huấn luyện chỉ bởi tiết kiệm, cả bộ dự đoán và đầu có thể diễn giải đều đọc một biểu diễn đã đóng băng, và phân tích gradient xác nhận mục tiêu phụ trợ không rò rỉ vào cả hai lộ trình kia.

![Hình A2: Định tuyến gradient qua bức tường detach.](figs/A2_gradient_decoupling.png)

*Hình A2. Định tuyến gradient qua bức tường detach (§3.7). KL → backbone; BCE dự đoán liên kết → đầu tồn tại (`s_t1_pos`); CE chống-sụp-đổ → đầu `s_t1_cal` phân cấp; tự-nhất-quán độ-tin-cậy → ghép niềm-tin-theo-bước (mặc định tắt). Bức tường dừng mọi gradient Luồng-B tại `edge_h.detach()` — được kiểm chứng bằng gradient backbone bằng không trên cả 56 tensor.*

---

## 4. Động cơ phản thực / can thiệp

**Vì sao một bộ dự đoán liên kết thời gian nên có thể được thẩm vấn (interrogable).** Một người thực hành thường muốn hỏi *điều-gì-nếu (what-if)* — *nếu cặp này bị buộc đi vào suy giảm, xác suất cạnh của nó sẽ rớt bao xa?* Đây là những câu hỏi can thiệp, $P(\text{edge} \mid \mathrm{do}(\cdot))$, không phải $P(\text{edge} \mid \text{observed})$, và các baseline chủ đạo không thể trả lời chúng: JODIE, TGAT, CAWN, TGN và GraphMixer sinh ra một điểm số từ một embedding bị quấn chặt (entangled) mà không có trạng thái được phơi bày, có kiểu ngữ nghĩa (semantically-typed) để can thiệp, và việc nhiễu loạn đầu vào chỉ là tương quan (correlational) mà không có bảo đảm rằng nó ánh xạ tới một sự kiện vòng đời có nghĩa. Các bộ giải thích hậu kỳ như GNNExplainer [Ying et al., 2019] khớp một surrogate thay vì phơi bày một cơ chế mà mô hình tính toán xuyên qua. Bộ đọc vòng đời của SR-GNN lấp khoảng trống này: trạng thái được giải mã được sinh bởi một toán tử chuyển tường minh, đã biết, nên *bản thân mô hình* là có thể can thiệp và `do(·)` là một chỉnh sửa thực, không phải một dò xét hộp-đen — và điều này đi kèm miễn phí trên cùng thao tác detach vốn mua lại lợi thế inductive (§3.7), nên toàn bộ dàn thử nghiệm chạy với chi phí bằng không cho dự đoán.

**Hình thức luận: bộ đọc vòng đời như một SCM có thể can thiệp.** `s_t1_cal` là một phân phối năm-trạng-thái đích thực được sinh bởi một toán tử chuyển đã biết $T_{uv}$ (cơ sở $UV^\top$ hạng-thấp cộng cổng theo từng cặp $g(\phi_{uv})$ của §3.6) trên ma trận luật-nhân-quả $C$, mà các cổng của nó là hàm của các tác nhân (driver) có tên: $p_{\text{birth}}(n_{\text{prior}})$, $p_{\text{alive}}(\text{rate}, \text{staleness})$, $p_{\text{rising}}(\text{slope})$ (§3.3). Đây là một mô hình nhân quả cấu trúc theo nghĩa của Pearl [Pearl, 2009]: các tác nhân là các nguyên nhân ngoại sinh (exogenous), các cổng là các phương trình cấu trúc, và trạng thái được giải mã là biến nội sinh (endogenous) được bộ giải mã tồn tại đọc. Động cơ hỗ trợ hai chế độ: một **can thiệp tác nhân (driver intervention)** ($\mathrm{do}(\text{rate}/\text{slope}/\text{staleness})$) ghi đè một thống kê có tên và tái lan truyền qua các cổng (mềm, tôn trọng cơ chế đã học), còn một **can thiệp trạng thái (state intervention)** $\mathrm{do}(\text{state}{=}s)$ thay thế bản rút đã gating bằng một trạng thái one-hot cố định và lan truyền nó qua bộ giải mã tồn tại với các thống kê thượng nguồn giữ cố định (cứng, cắt đứt trạng thái khỏi các nguyên nhân của nó). Trong cả hai, động cơ trước hết tái dựng đường nền (baseline) cổng một cách chính xác (phần dư $1\mathrm{e}{-16}$), nên bất kỳ thay đổi được đo nào cũng quy được hoàn toàn cho riêng can thiệp. Mọi phép đo đều trên các cặp CoEdit thực (config B, đã sửa P1, $N = 12000$), và toàn bộ dàn thử nghiệm được **kiểm chứng trên ba seed** ({42, 1, 7}, mỗi seed huấn luyện đến dump config-B `cbON`): thang tồn tại chính xác theo cấu trúc trên mọi seed, và các kết quả quỹ đạo cùng đáp-ứng-theo-liều dưới đây được báo cáo dưới dạng trung bình ± độ lệch chuẩn ba seed.

**Phản thực tồn tại (cầu nối luận điểm).** `do(state = DEATH)` đưa xác suất tồn tại được dự đoán về 0, với một sự rớt ở **ít nhất 99% các cặp trên mọi seed** (trung bình $\Delta \approx -0.52$); `do(state = REINFORCE)` và `do(BIRTH)` đưa nó về 1, với một sự tăng ở **ít nhất 99% trên mọi seed**. Toàn bộ thang can thiệp là đơn điệu và chính xác, đọc trực tiếp các trọng số của bộ giải mã tồn tại, và giữ **giống hệt trên cả ba seed**: DEATH 0.0 < IDLE 0.1 < DECAY 0.3 < đường nền quan sát $\approx 0.52$ < BIRTH = REINFORCE 1.0. Đẳng thức BIRTH = REINFORCE đến từ việc bộ giải mã tồn tại đối xử cả hai như "cạnh có mặt" với trọng số 1; thứ tự chính xác là các trọng số tồn tại, không phải một khớp thực nghiệm. Một kiểm tra null xác nhận tính đặc hiệu (specificity): `do(noop)` cho ra $\Delta = 0$ **chính xác trên mọi seed**. **Tính đảo ngược là chính xác:** can thiệp xuôi (forward) có cắn (max $|\Delta\, p_{\text{edge}}| \approx 0.52$); sau khi hoàn tác (undo), max $|\Delta\, p_{\text{edge}}|$ và max $|\Delta\, \text{state-dist}|$ là chính xác 0.00e+00 **trên cả ba seed**, xác nhận động cơ khôi phục *chính xác* phép tính trước-can-thiệp chứ không phải một xấp xỉ của nó — một đặc tính mà một bộ giải thích surrogate không thể cung cấp.

![Hình 3: Thang phản thực tồn tại trên N=12000 cặp CoEdit.](figs/fig3_counterfactual_ladder.png)

*Hình 3. Thang phản thực tồn tại trên các cặp CoEdit thực (N=12000, config B). `do(state)` buộc từng trạng thái vòng đời và đọc P(edge) được dự đoán: DEATH 0.0 < IDLE 0.1 < DECAY 0.3 < đường nền quan sát 0.52 < BIRTH = REINFORCE 1.0. Tác động là chính xác (trọng số tồn tại one-hot), đơn điệu, đúng-dấu, và đảo ngược chính xác trên cả ba seed {42, 1, 7}; do(DEATH) hạ P(edge) cho ≥99% các cặp trên mọi seed, do(noop) cho Δ=0 chính xác.*

**Đáp ứng theo liều (dose-response) và tính đúng-dấu trên các tác nhân thực.** Các can thiệp tác nhân dịch chuyển phân phối trạng thái một cách đơn điệu và theo chiều đúng về mặt vật lý, không có đảo dấu, và các dấu của tác động **ổn định trên cả ba seed**: nâng tỷ số tốc độ làm tăng $P(\text{REINFORCE})$ ($\Delta = +0.028 \pm 0.004$) và giảm $P(\text{DEATH})$ ($\Delta = -0.068 \pm 0.007$); nâng độ dốc tăng (rising-slope) làm tăng $P(\text{REINFORCE})$ ($\Delta = +0.489 \pm 0.003$, tác động lớn nhất và chặt nhất) và giảm $P(\text{DECAY})$; nâng true-occurrence làm giảm $P(\text{BIRTH})$ ($\Delta = -0.0065$, đúng theo việc hạ bậc các cặp mới-tinh khi sự tái diễn tích lũy); nâng độ cũ (staleness) làm tăng $P(\text{DECAY})$ rồi $P(\text{DEATH})$. Đáp ứng theo liều là sạch, đơn điệu, và ổn định ba seed cả về dấu lẫn độ lớn trên các tác nhân CoEdit thực (rate, slope, true-occurrence) — bằng chứng trục-alive làm nền cho các kết quả quỹ đạo dưới đây.

**Phản thực quỹ đạo trên các cặp thực.** Vượt ra ngoài việc buộc một trạng thái, động cơ trả lời được liệu *quỹ đạo tương lai* của một cặp có thể được chuyển hướng hay không. Trên các cặp được giải mã là DECAY, một $\mathrm{do}(\text{slope} = +)$ tổng hợp lật trạng-thái-kế-tiếp DECAY→REINFORCE cho **0.9999 ± 0.0002 trên ba seed** (theo từng seed 1.0 / 1.0 / 0.9996) — cơ chế tăng được nối dây, có phản hồi và ổn định theo seed, không phải một sự trùng hợp đơn-seed. Trên các cặp được giải mã là REINFORCE (tập con tái diễn, $\text{true\_occ}\ge 2$), các can thiệp theo hướng-giết phân rã sạch theo liều, và phân rã đó **ổn định trên ba seed**. Mỗi tác nhân đơn lẻ bị đẩy về giá trị chết chỉ là *một phần*: một $\mathrm{do}(\text{rate} = \text{dead})$ cô lập đưa REINFORCE→DEATH cho **0.60 ± 0.03** các cặp (theo từng seed 0.574 / 0.588 / 0.629), và một $\mathrm{do}(\text{staleness} = \text{high})$ cô lập cho **0.48 ± 0.01** (0.477 / 0.479 / 0.489). Đẩy *tất cả* các tác nhân trục alive chết cùng lúc — $\mathrm{do}(\text{rate}{=}\text{dead}, \text{slope}{=}\text{falling}, \text{staleness}{=}\text{high})$ — là **quyết định**: REINFORCE→DEATH cho **0.999 ± 0.002** (1.0 / 1.0 / 0.997). Phân rã theo liều là sạch: mỗi tác nhân góp một phần của cú giết và hợp lại thì quyết định, đối xứng theo hướng-giết với cú lật theo hướng-tăng ở trên — nên $T_{uv}$ hỗ trợ kiểm soát quỹ đạo **hai chiều**, cả hai chiều giờ đều đã kiểm chứng trên ba seed. Những can thiệp trên các tác nhân thực của các cặp thực này cho thấy $T_{uv}$ mang động lực học vòng đời đích thực, có thể chuyển hướng được, không phải một nhãn theo từng cặp tĩnh (ví dụ minh họa: cặp 3178→7437, Hình 6, §3.3). Các con số tiêu đề cũ 83–86% (rate) và 100% (staleness) đến từ một dump đơn-seed cũ **không** tái lập được trên npz `cbON` ba-seed nhất quán; chúng tôi thay chúng bằng các con số ba-seed đo lại ở trên, và đây là các con số được dùng (dàn rate/staleness, `fsm_intervene.py`, tập con tái diễn; nguồn `cf_kill_REINFORCE_3seed.json`).

**Tính AP-trung-tính (interrogability đi kèm miễn phí trên detach).** Mọi kết quả ở trên được đọc ra từ `s_t1_cal`, ở phía đã detach của bức tường gradient §3.7; bộ giải mã tồn tại phản thực là một bản đọc đóng băng, không phải bộ dự đoán được chấm điểm. Chạy toàn bộ dàn thử nghiệm làm nhiễu loạn đường AP đúng **chính xác $\Delta = 0$** (giống hệt từng byte khi tắt động cơ). Cú chính-xác-bằng-không này là một phát biểu *tại thời điểm đánh giá* trên một mô hình đã huấn luyện **đóng băng**: bản đọc phản thực không bao giờ chạm vào logit được chấm điểm, nên bật/tắt động cơ để lại mọi tensor liên quan đến AP giống hệt từng byte. (Điều này khác với việc *huấn luyện* mô hình có so với không có `hier_causal_policy`, vốn chỉ AP-trung-tính trong phạm vi nhiễu seed — không phải giống-hệt-byte — vì thêm op chính sách làm dịch chuyển luồng RNG huấn luyện; xem §6.2.) Không baseline nào phơi bày một vòng đời có thể can thiệp, nên tính có-thể-thẩm-vấn này không có đối ứng trong các bộ dự đoán liên kết thời gian trước đây và không tốn gì cho chúng tôi.

**Phạm vi trung thực.** SCM là sạch và trung thực trên *trục alive* (rate / recurrence → alive → DEATH/REINFORCE), và toàn bộ dàn phản thực — thang tồn tại, các dấu đáp-ứng-theo-liều, cú lật tăng DECAY→REINFORCE, và phân rã giết REINFORCE→DEATH — giờ **đã được kiểm chứng trên ba seed trên CoEdit** ({42, 1, 7}); thang tồn tại chính xác theo cấu trúc trên mọi seed, còn đáp-ứng-theo-liều, lật-slope và các kết quả giết mang độ lệch chuẩn ba seed chặt. Các can thiệp giết theo tác nhân đơn lẻ chỉ là *một phần* nhưng ổn định theo seed (rate-một-mình 0.60 ± 0.03, staleness-một-mình 0.48 ± 0.01 → DEATH), còn can thiệp tất-cả-tác-nhân-chết là *quyết định* (0.999 ± 0.002, ba seed); cùng với cú lật ngược chiều DECAY→REINFORCE (0.9999) điều này cho kiểm soát quỹ đạo hai chiều, cả hai chiều đều khóa-ba-seed. *Trục rising* (slope → phân chia REINFORCE-so-với-DECAY) được nối dây nhân quả — $\mathrm{do}(\text{slope} = +)$ tổng hợp lật DECAY→REINFORCE cho $0.9999 \pm 0.0002$ các cặp được dò qua các seed — nhưng **suy biến (degenerate) trên CoEdit**, bởi `slope_rel` của CoEdit về cơ bản luôn âm (không có độ dốc tốc độ chỉnh sửa dương bền vững), nên DECAY→REINFORCE là không đạt được từ các tác nhân CoEdit thực dù cơ chế tồn tại; trục staleness cũng tương tự chỉ được vận dụng trên các phép tiêm tổng hợp. Vận dụng các trục này trên CoEdit thực sẽ đòi hỏi một tín hiệu nhịp điệu bản địa của CoEdit và/hoặc một loss tính-nhất-quán-phản-thực (counterfactual-consistency loss) — công việc tương lai — và chúng tôi không tuyên bố trục rising là một phản thực đã được kiểm chứng trên dữ liệu CoEdit thực.

---

## 5. Độ tin cậy theo tính-nhất-quán-nhân-quả (phạm vi trung thực)

Chúng tôi cũng khảo sát liệu mô hình có thể tự gắn cờ (flag) các dự đoán độ-tin-cậy-thấp của *chính nó* qua một tín hiệu **tính-nhất-quán-nhân-quả theo chuỗi-bước (walked-chain causal-coherence)** chạy song song (và không bao giờ che) đường dự đoán. Một niềm tin (belief) theo từng cặp $b_t$ được mang bởi toán tử đã học $T_{uv}$ chiếu lên tia khả-thừa-nhận-nhân-quả, được ghép nhẹ với phép đo pha-quan-sát; tính nhất quán $c_t \in [0,1]$ là độ tương hợp giữa dự đoán trạng-thái-kế-tiếp tự do của mô hình và niềm tin theo-bước này. Cờ `causal_confidence` mặc định tắt và giống hệt từng byte khi tắt, nên nó không bao giờ làm nhiễu loạn AP của config B.

Trên ba seed (CoEdit, grounded-init), AP được bảo toàn và $c_t$ phân tách sạch theo kết cục luật-nhân-quả: các dự đoán *tuân theo (following)* luật mang tính nhất quán trung bình 0.891 ± 0.042, các dự đoán *vi phạm (violating)* luật 0.216 ± 0.174 — một tín hiệu trải rộng tốt, không bị sụp đổ, với toàn bộ miền giá trị $[0, 1]$.

![Hình 5: Tín hiệu tính-nhất-quán-nhân-quả c_t theo kết cục (ba seed, grounded-init).](figs/fig5_causal_coherence.png)

*Hình 5. Tính-nhất-quán-nhân-quả c_t theo kết cục (CoEdit, grounded-init, ba seed, độ lệch chuẩn mẫu): nhất-quán-với-luật 0.891 ± 0.042 so với vi-phạm-luật 0.216 ± 0.174. Một thước đo tự-nhất-quán mang tính tư vấn (advisory) — nó gắn cờ các vi phạm luật của chính mô hình, không phải lỗi dự đoán ngoại tại (§5).*

**c_t là gì — và không là gì.** Tính nhất quán thấp dự đoán sự vi phạm luật-nhân-quả của *chính* mô hình gần như hoàn hảo và ổn định: AUC = **0.9985 ± 0.0015** trên ba seed. Nhưng đây là **tự-nhất-quán (self-consistency)**, không phải sự thật ngoại tại: nó đo liệu một dự đoán có tuân theo các luật vòng đời của chính mô hình hay không, đó là bằng chứng vòng vo (circular) về tính đúng đắn — một mô hình tự tin và nhất quán mắc cùng một lỗi sẽ chấm điểm cao. Khi chúng tôi kiểm thử liệu tính nhất quán thấp có dự đoán *các lần trượt dự đoán thực tế* (mục tiêu ngoại tại `posMiss10`) hay không, AUC là **0.405 ± 0.484** — phụ thuộc seed một cách dữ dội (0.949 / 0.245 / 0.021). Một kết quả đơn-seed đầy hứa hẹn (0.949) đã **không** tái lập được qua seed 1 và 7. Do đó chúng tôi **rút lại** mọi tuyên bố rằng $c_t$ là một bộ dự báo lỗi hay tính đúng đắn và chỉ báo cáo nó như một thước đo nhất quán nội tại ổn định. Chúng tôi xem đây là cách đọc trung thực về bằng chứng và gắn cờ khoảng trống này như một vấn đề mở: biến tự-nhất-quán thành một bộ dự báo lỗi đã được hiệu chỉnh sẽ đòi hỏi một tín hiệu giám sát ngoại tại mà chúng tôi chưa xây dựng.

![Hình A5: Cơ chế độ-tin-cậy-nhân-quả (lớp phủ tính nhất quán tư vấn c_t).](figs/A5_causal_confidence_overlay.png)

*Hình A5. Cơ chế độ-tin-cậy theo tính-nhất-quán-nhân-quả. Dự đoán trạng-thái-kế-tiếp tự do của mô hình được phủ lên niềm tin theo-bước $b_t$; sự tương hợp của chúng là tính nhất quán tư vấn $c_t$. Mặc định tắt và giống hệt từng byte khi tắt; nó đo tự-nhất-quán, không phải lỗi ngoại tại (§5).*

---

## 6. Thực nghiệm

### 6.1 Liên-dataset, khớp-giao-thức, ba seed

Mọi mô hình — SR-GNN và sáu baseline — chạy qua **cùng** harness train/eval (`experiments/train.py:run_epoch`), cùng các split theo trình tự thời gian 70/15/15, và cùng pool negative đã-được-kiểm-toán-rò-rỉ (§6.3 xác nhận test AP không phải 1.0). Pool được xây dựng công bằng theo từng giao thức: các negative transductive rút từ các cặp seen→seen, inductive từ pool node-chưa-thấy (ind→ind), nên một positive inductive không bao giờ bị chấm điểm so với một negative bất khả thi một cách tầm thường. AP là `average_precision_score` của sklearn trên pool đó, đồng nhất cho mọi mô hình; các chỉ số là trung bình ± std trên các seed {42, 1, 7}. SR-GNN ở đây là **config B**, được tinh chỉnh chỉ trên CoEdit.

![Hình 1: AP inductive CoEdit, SR-GNN (config B) so với sáu baseline khớp-giao-thức, ba seed. SR-GNN decoupled đạt 0.9885 ± 0.0035, +13.5 điểm so với baseline tốt nhất (TGAT, 0.853). Cột là trung bình; thanh sai số là std mẫu trên các seed {42, 1, 7}.](figs/fig1_coedit_headline.png)

*Hình 1. AP inductive CoEdit: SR-GNN (config B) so với sáu baseline khớp-giao-thức, ba seed (std mẫu). SR-GNN đạt 0.9885 ± 0.0035, +13.5 điểm so với baseline tốt nhất (TGAT, 0.853). CoEdit là benchmark trưng bày; biên độ inductive ở đây là kết quả tiêu đề.*

**Bảng 1 — AP inductive (trung bình ± std, 3 seed).**

| Model | CoEdit ind-AP | Wikipedia ind-AP | MOOC ind-AP |
|---|---|---|---|
| **SR-GNN (config B)** | **0.9885 ± 0.0035** | 0.9959 ± 0.0014 | **0.9978 ± 0.0013** |
| JODIE | 0.8147 ± 0.0942 | 0.9860 ± 0.0029 | 0.9942 ± 0.0018 |
| TGAT | 0.8530 ± 0.0012 | **0.9981 ± 0.0013** | 0.9763 ± 0.0134 |
| CAWN | 0.7825 ± 0.0452 | 0.9877 ± 0.0062 | 0.8924 ± 0.1663 |
| TGN | 0.6349 ± 0.0065 | 0.8637 ± 0.0459 | 0.9819 ± 0.0048 |
| DyRep | 0.6218 ± 0.0119 | 0.6314 ± 0.0550 | 0.6675 ± 0.2824 |
| GraphMixer | 0.6232 ± 0.0247 | 0.7380 ± 0.0770 | 0.9827 ± 0.0055 |

**Bảng 2 — AP transductive (trung bình ± std, 3 seed), SR-GNN và các baseline đáng chú ý.**

| Model | CoEdit trans-AP | Wikipedia trans-AP | MOOC trans-AP |
|---|---|---|---|
| **SR-GNN (config B)** | **0.9985 ± 0.0004** | **0.9993 ± 0.0002** | **0.9988 ± 0.0002** |
| JODIE | 0.9657 | 0.9954 | 0.9919 |
| TGAT | 0.8690 | 0.6578 | 0.6174 |
| CAWN | 0.8802 | 0.9861 | 0.9727 |

**Đọc bảng.**
- **CoEdit** là benchmark phân biệt: SR-GNN là #1 theo cả hai chiều, và biên độ inductive so với baseline tốt nhất (TGAT, 0.853) là **+13.5 điểm** — kết quả tiêu đề. CoEdit là phi-lưỡng-phân (non-bipartite) (cả hai đầu mút đều mang một vòng đời), nên biểu diễn theo từng cặp có nhiều điều để nói nhất, và các baseline trải rộng (0.62–0.85), khác với các dataset đã bão hòa.
- **Wikipedia:** SR-GNN là #1 transductive (0.9993) và #2 inductive (0.9959 so với TGAT 0.9981). Nhưng chiến thắng inductive của TGAT đi kèm với một AP transductive *sụp đổ* (0.6578) — một sự kỳ quặc về đặc trưng, không phải sức mạnh toàn diện. SR-GNN là mô hình toàn diện tốt nhất ở đây, chính là đặc tính mà bảng hai-giao-thức được thiết kế để làm hiện rõ.
- **MOOC:** SR-GNN là #1 theo cả hai chiều, nhưng dataset gần-bão-hòa (nhiều mô hình vượt 0.98 inductive), nên chúng tôi không rút ra kết luận mạnh nào ngoài "cạnh tranh ở mức trần (ceiling)".

![Hình 4: SR-GNN (config B) so với baseline khớp-giao-thức tốt nhất theo từng dataset, AP transductive và inductive, ba seed. CoEdit là một chiến thắng rõ ràng của SR-GNN; Wikipedia và MOOC ngang ngửa (TGAT nhỉnh hơn SR-GNN một chút về inductive trên Wikipedia).](figs/fig4_cross_dataset.png)

*Hình 4. Tóm tắt liên-dataset: SR-GNN (config B) so với baseline tốt nhất theo từng dataset, AP transductive và inductive, ba seed. SR-GNN là chiến thắng rõ ràng trên CoEdit và tốt-nhất-hoặc-đồng-tốt trên Wikipedia/MOOC (trên Wikipedia inductive, 0.9981 của TGAT nhỉnh hơn 0.9959 của SR-GNN một chút, nhưng AP transductive của TGAT sụp đổ xuống 0.658). CoEdit là benchmark trưng bày; Wikipedia/MOOC ngang ngửa ở mức trần.*

### 6.2 Ablation tách rời (thí nghiệm lõi)

Để rõ ràng về cái được so sánh: **config B là mô hình đầy đủ** (backbone đã detach + toán tử đa tín hiệu + bộ đọc vòng đời phân cấp + động cơ phản thực + chính sách nhân quả), và hai arm dưới đây là *các ablation lược bỏ một cơ chế khỏi nó*, không phải các mô hình cạnh tranh. **Thí nghiệm tách rời lõi** là B so với C: cùng code, cùng đầu, cùng dữ liệu; thay đổi duy nhất là liệu link-prediction loss có chảy vào backbone hay không.

| Arm (CoEdit, ba seed {42,1,7}) | thiết kế | ind-AP | trans-AP |
|---|---|---|---|
| **B — decoupled (mô hình đầy đủ)** | correct_decoupled | **0.9885 ± 0.0035** | 0.9985 |
| C — end-to-end (decoupling tắt) | correct | 0.7672 ± 0.0107 | 0.9609 |
| A — không-vòng-đời, không-tinh-chỉnh | toán tử v3 + bộ đọc phẳng, đã detach | 0.928 ± 0.0043 | 0.9912 |

![Hình 2: ablation tách rời, AP inductive CoEdit. B 0.9885, C 0.767, A 0.928 (ba seed); detach đáng giá +22.1 điểm.](figs/fig2_decoupling_ablation.png)

*Hình 2. Ablation tách rời, AP inductive CoEdit, ba seed {42, 1, 7}: B (mô hình đầy đủ, decoupled, 0.9885 ± 0.0035) so với C (end-to-end, decoupling tắt, 0.7672 ± 0.0107) so với A (SR-GNN không-vòng-đời, ablation không-tinh-chỉnh cô lập cơ chế detach, 0.928 ± 0.0043). Các arm A và C là ablation của mô hình đầy đủ B, không phải các mô hình cạnh tranh. Ghép end-to-end làm sụp đổ AP inductive; detach đáng giá **+22.1 ± 1.4 điểm** (giao thức config-B so với design=correct, mọi seed trên 20 điểm: B−C theo từng seed = [0.206, 0.225, 0.234]). Ablation không-vòng-đời (A) đã đánh bại mọi baseline (0.928 ± 0.0043, +7.5 pp so với TGAT). (Chiều cao cột thể hiện ind-AP trung bình seed; PNG nền được vẽ từ các điểm seed-42 trước đó và sẽ được vẽ lại tại C = 0.767.)*

Tách rời backbone đáng giá **+22.1 ± 1.4 điểm** AP inductive (B 0.9885 − C 0.7672) qua ba seed dưới giao thức config-B so với design=correct hiện tại, với khoảng cách theo từng seed [0.206, 0.225, 0.234] đều trên 20 điểm và các khoảng theo từng seed không chồng lấn. Một A/B thô hơn, sớm hơn (decoupled so với end-to-end, ba seed, giao thức P0) đặt cùng hiệu ứng ở **+9.4 điểm tuyệt đối**; chúng tôi báo cáo cả hai cùng các giao thức của chúng và không trộn lẫn — khoảng cách lớn hơn phản ánh đầu config-B đã được tinh chỉnh đầy đủ, khoảng cách nhỏ hơn là đầu A/B nguyên bản. Then chốt là, ghép end-to-end hầu như không dịch chuyển AP transductive (0.9985 → 0.9609) trong khi AP inductive *sụp đổ* (0.9885 → 0.7672): nó không chỉ thất bại trong việc giúp ích theo inductive — nó chủ động phá hủy sự tổng quát hóa inductive trong khi để lại con số transductive gần như nguyên vẹn. Chúng tôi đọc điều này là: link loss, khi được trao quyền ghi vào backbone, tái định hình biểu diễn về danh tính node huấn luyện (tối đa hóa xếp hạng transductive) với cái giá trực tiếp là các động lực học theo từng cặp tổng quát có thể chuyển giao sang các node chưa thấy. Detach loại bỏ động cơ đó, và đó là lý do cùng một đầu, trên một backbone đã đóng băng, tổng quát hóa được.

**Chỉ riêng tách rời — không phải bộ máy vòng đời — vượt qua các baseline (arm A).** Arm A là ablation *không-vòng-đời, không-tinh-chỉnh*: toán tử v3 đã detach được đọc qua một bộ đọc phẳng đơn thuần, không có CE chống-sụp-đổ, không có giải mã phân cấp, không có `causal_batch`, và không có tinh chỉnh theo từng dataset. Nó không phải một mô hình thứ hai mà là sàn (floor) cô lập cơ chế tách rời. Ngay cả ở sàn đó, SR-GNN đạt **0.928 ± 0.0043** AP inductive trên ba seed — đã **+7.5 điểm so với baseline tốt nhất** (TGAT, 0.853) trên giao thức đồng nhất (ML integrity audit, 2026-06-06). Đây là bằng chứng thứ cấp rằng chiến thắng inductive được gánh bởi *chỉ-detach (detach-alone)*, không phải bởi bộ máy vòng đời. Thêm **+6 điểm** từ arm A tới config B (0.928 → 0.9885) là cái mà giám sát vòng đời thêm vào — và chính giám sát đó mang lại lớp diễn giải và phản thực trung thực, có thể can thiệp (§4, §5) với chi phí AP bằng không (§3.4). Hai khoảng cách phân tách sạch sẽ: detach mua sự tổng quát hóa inductive đánh bại baseline; giám sát vòng đời mua tính diễn giải cộng thêm một biên độ AP nữa, miễn phí trên đường được chấm điểm.

**Hai ablation thêm.**
- **`causal_batch` (bản sửa đọc-trước-khi-ghi, §3.2):** trong config B đầy đủ, BẬT 0.9885 so với TẮT 0.9312 inductive (**+5.7 ± 0.2 pp**, ba seed; trans +0.65 pp), xác nhận các thống kê theo từng cặp đã-sụp-đổ trước đó là quan trọng (load-bearing). Cùng dấu trong thiết lập không-vòng-đời được lược (A/B đơn-seed 0.7907 so với 0.7462, job 5467100).
- **`hier_causal_policy` (mặt nạ nhân quả mềm, §3.5):** ba seed {1, 7, 42}, huấn luyện BẬT so với TẮT, Δ inductive theo từng seed = +1.0e-4 / +9.3e-4 / −1.5e-3 (tối đa $|\Delta| = 1.5\mathrm{e}{-3}$, trung bình ≈ −1.7e-4, lẫn-dấu) so với độ lệch chuẩn seed $\pm 3.5\mathrm{e}{-3}$ (job 5511229). **AP-trung-tính trong phạm vi nhiễu seed** — chứng minh bất biến điểm số của §3.4 bảo đảm không có hiệu ứng *có hệ thống* (đầu ra chính sách không bao giờ đưa vào loss AP); phần dư ~1e-3 là jitter RNG huấn luyện, không phải khớp giống-hệt-byte. Chính sách mua tính diễn giải (về mặt thống kê) miễn phí.

### 6.3 Kiểm toán liêm chính (Integrity audit)

Vì kết quả tiêu đề dựa trên các so sánh liên-giao-thức, chúng tôi đã kiểm toán nó độc lập (2026-06-06). Các phát hiện: (1) kết quả không-vòng-đời-vượt-baseline (arm A, 0.928) là thực, không phải tạo tác — các con số baseline cũ so với mới giống hệt trong phạm vi nhiễu GPU tại các seed khớp, và bước 0.871 → 0.928 là một khác biệt *config* (toán tử theo từng cặp v3 trên một bộ đọc phẳng), không phải một sự dịch chuyển đánh giá; (2) đánh giá trước-cập-nhật là không-rò-rỉ (việc tái-gating chống-rò-rỉ, job 5450095, kéo test AP khỏi 1.0 vào dải v2, loại trừ một rò rỉ nhãn cùng-batch); (3) AP là một thường trình sklearn không phụ thuộc mô hình trên cùng pool negative cho mọi mô hình, nên không mô hình nào có một bộ đánh giá riêng. Chúng tôi báo cáo cuộc kiểm toán vì tính đáng tin của kết quả phụ thuộc vào việc giao thức là công bằng và có thể được reviewer kiểm tra.

### 6.4 Tính trung thực của vòng đời

Trên bộ đọc phân cấp, trạng thái DECAY trung gian cuối cùng cũng mang khối argmax thực: trên config cuối cùng (dump faithfulness seed-42, $N=12000$), DECAY là argmax của `s_t1_cal` cho **47.8%** các cặp (5737/12000), so với **0.04%** (5/12000) dưới bộ đọc phẳng trên cùng dump — một mức tăng **~1147×**. Các luồng đang-giảm-nhưng-vẫn-hoạt-động thắng argmax DECAY khi còn sống, các cặp im-lặng-bền-vững đi tới DEATH, các cặp mới tới BIRTH. Trạng thái được giải mã bám theo nhịp điệu của *chính* mỗi cặp (Spearman $\rho(\texttt{p\_decay\_cal}, \texttt{slope\_rel}) \approx -0.59$ trên tập con tái diễn $\text{true\_occ}\ge 2$, $n=9157$, đúng dấu, $p < 10^{-300}$) — trung thực về mặt xác suất và giờ đây cả về argmax. Tính trung thực ở đây không được khẳng định bởi một bộ giải thích phụ trợ; nó là cùng một `s_t1_cal` mà CE chống-sụp-đổ giám sát, và §3.4 chứng minh nó là chính xác đại lượng mà các hình vẽ.

### 6.5 Bộ đọc vòng đời có ý nghĩa trên nhiều dataset, không phải một artifact của CoEdit

Một lo ngại tự nhiên là vòng đời năm-trạng-thái chỉ là một artifact của CoEdit và sẽ sụp đổ hoặc trông giống hệt nhau trên dữ liệu khác. Không phải vậy. Chúng tôi chạy phân tích faithfulness config-B trên tập con tái diễn ($\text{true\_occ}\ge 2$) của ba dataset và đọc phân phối argmax của `s_t1_cal` trên các trạng thái hoạt động. Trên mỗi dataset, phân phối là không-suy-biến và *hình dạng của nó khớp với động lực học riêng* của dataset đó:

**Bảng 3 — Phân phối argmax vòng đời trên các trạng thái hoạt động (faithfulness config-B, tập con tái diễn, seed 42).**

| Dataset | REINFORCE | DECAY | DEATH | Hình dạng |
|---|---|---|---|---|
| CoEdit | 0.35 | **0.62** | 0.02 | nặng-DECAY — tàn chậm, hiếm chết hẳn |
| Wikipedia | **0.42** | 0.43 | 0.16 | cân bằng, đủ vòng đời tới DEATH (entropy 1.31) |
| MOOC | **0.62** | 0.08 | 0.30 | nặng-BIRTH, thoáng qua — sinh, hoạt động, rồi chết |

Bộ đọc *thích nghi* với từng miền: chỉnh sửa trên CoEdit tàn chậm nên khối tập trung ở DECAY với hiếm khi chết hẳn; Wikipedia chạy một vòng đời cân bằng đầy đủ đi tới DEATH; hoạt động MOOC là thoáng qua (chạm một đơn vị, hoạt động ngắn, rồi dừng hẳn), cho một hình dạng nặng-BIRTH, đuôi-DEATH với ít DECAY. Không dataset nào suy biến — không trạng thái đơn lẻ nào nuốt hết khối — và các hình dạng theo động lực học riêng của từng miền chứ không phải một khuôn mẫu toàn cục, nên bộ đọc vòng đời là một diễn giải có ý nghĩa, điều-kiện-theo-dữ-liệu, **không phải một artifact đặc thù của CoEdit**.

Chúng tôi nói rõ phạm vi: bằng chứng vòng đời liên-dataset này là một *kiểm tra sanity faithfulness* đơn-seed (seed 42), xác lập rằng hình dạng bộ đọc là có ý nghĩa và phù-hợp-dataset trên cả ba dataset; nó **không phải** một nghiên cứu liên-dataset ba-seed đầy đủ. Ngược lại, dàn phản thực *thì* đã khóa ba-seed đầy đủ, trên CoEdit (§4).

---

## 7. Phân tích

**Vì sao tách rời giúp ích theo inductive.** Cơ chế là kiểm soát truy cập (access control): một backbone có quyền truy cập gradient vào link loss được tưởng thưởng cho các đặc trưng danh tính có tính dự đoán theo transductive nhưng là khối lượng chết trên một node chưa thấy, nên các mô hình end-to-end phải trả một khoản thuế inductive. Backbone của SR-GNN không bao giờ thấy gradient liên kết (§3.7), nên nó *không thể* học các lối tắt danh tính. Khoảng cách ablation +22.1 ± 1.4 pp (ba seed) đo khoản thuế đó, và mặt transductive gần-phẳng của nó (0.9985 → 0.9609) cho thấy nó hầu như chỉ được trả theo inductive, đúng như câu chuyện lối-tắt-danh-tính dự đoán. Chúng tôi định vị đây là một giả thuyết cơ chế với một dự đoán có thể phủ định (falsifiable) — lợi thế nên tăng theo độ mới (novelty) của tập inductive — chứ không phải một định lý; một dò xét danh tính (identity-probe) theo từng split là thí nghiệm tiếp theo tự nhiên.

**Vì sao tính diễn giải và phản thực đến miễn phí.** Sự phân chia hai-đầu (§3.4) giữ bộ giải mã vòng đời nằm ngoài đường được chấm điểm với tính bất biến điểm số chính-xác-bằng-không, nên sự đánh đổi diễn-giải-độ-chính-xác thường thấy [Rudin, 2019] không áp dụng: trạng thái ký hiệu là một bản *đọc* trung thực của chính backbone của bộ dự đoán với chi phí AP bằng không. Cùng đặc tính đó làm cho phản thực mang tính bản địa thay vì hậu kỳ — `do(state)` chỉnh sửa một toán tử chuyển tường minh với các hiệu ứng đơn điệu, đảo ngược được, đúng-dấu (§4), không phải một khớp surrogate. Ranh giới được báo cáo, không bị che giấu: trên CoEdit trục rising suy biến, nên một trong bốn trục nhân quả được nối dây nhưng không được vận dụng trên các cặp thực.

---

## 8. Hạn chế (Limitations)

Chúng tôi nêu rõ những điều này; một số được gắn cờ trong chính các ghi chú kiến trúc của chúng tôi.

1. **Được-tinh-chỉnh-trên-CoEdit.** Việc tinh chỉnh config B (giải mã phân cấp, decol_hier_v2, các thiết lập $\lambda$) được chọn trên CoEdit. Wikipedia và MOOC chạy cùng config mà không tinh chỉnh lại; biên độ inductive tiêu đề tập trung ở CoEdit. Chúng tôi không tuyên bố một +13.5 phổ quát.
2. **Phạm vi vòng đời: có ý nghĩa trên ba dataset, phản thực khóa đầy đủ trên CoEdit.** Bộ đọc vòng đời *không* phải artifact của CoEdit — phân phối argmax của nó là không-suy-biến và phù-hợp-dataset trên CoEdit, Wikipedia, và MOOC (§6.5, lần lượt nặng-DECAY / cân-bằng / thoáng-qua), nhưng bằng chứng liên-dataset đó là một kiểm tra sanity faithfulness đơn-seed (seed 42), không phải một nghiên cứu liên-dataset ba-seed đầy đủ. Dàn phản thực đã khóa ba-seed đầy đủ, nhưng chỉ trên CoEdit (§4). Riêng trong CoEdit, các trục nhân quả rising/staleness suy biến trên các tác nhân của CoEdit, nên SCM chỉ được *vận dụng* đầy đủ trên trục alive.
3. **Độ tin cậy là tự-nhất-quán, không phải dự đoán lỗi.** Như chi tiết ở §5, tín hiệu nhất quán dự đoán ổn định các vi phạm luật của chính mô hình (AUC 0.9985) nhưng *không* dự đoán đáng tin các lỗi dự đoán thực tế (AUC 0.405 ± 0.484 trên các seed). Chúng tôi rút lại tuyên bố dự-đoán-lỗi đơn-seed.
4. **Mùi cấu trúc trong chính sách nhân quả.** Bộ tích lũy `ever_alive` là phi-Markov và nằm ngoài ma trận chuyển không-bộ-nhớ $C$; bảo đảm chết-trước-khi-sinh được thực thi bởi một cổng riêng. Với một observer tự-tin-nhưng-sai, mặt nạ $C$ mềm có thể triệt tiêu một DEATH thực sự đúng.
5. **MOOC gần-bão-hòa**, nên kết quả #1 của nó là bằng chứng yếu; sự phân tách có nghĩa là trên CoEdit.
6. **Một synthetic chuyển-chế-độ (regime-switch) đã phủ định một giả thuyết.** Một synthetic điểm-thay-đổi (change-point) sạch cho thấy sự thích nghi theo từng cặp của SR-GNN *không* đánh bại CAWN trên các lát cắt sau điểm-thay-đổi; chúng tôi không tuyên bố một lợi thế chuyển-chế-độ. Lợi thế đã được kiểm chứng là bộ đọc inductive, không phải thích nghi chế độ nhanh hơn.
7. **Echo memory và một transition-CE học được đã được tuyên bố trước đây và đã được rút lại** — bộ chính quy backbone của mô hình hiện tại là một KL VAE/tiết kiệm, không phải một số hạng echo-memory; giám sát chuyển là CE chống-sụp-đổ trên `s_t1_cal`, không phải một số hạng transition-matrix CE riêng.
8. **Các kiểm tra đơn-seed còn lại.** Hai delta ablation tiêu đề (B so với C tách rời, +22.1 ± 1.4 pp; `causal_batch`, +5.7 ± 0.2 pp) *và* toàn bộ dàn phản thực — thang tồn tại, các dấu đáp-ứng-theo-liều, cú lật tăng DECAY→REINFORCE (0.9999 ± 0.0002), và phân rã giết REINFORCE→DEATH (tác nhân đơn lẻ chỉ một phần: rate 0.60 ± 0.03, staleness 0.48 ± 0.01; tất-cả-tác-nhân-chết quyết định 0.999 ± 0.002), §4 — nay đã được khóa ba-seed trên CoEdit. Kiểm tra AP-trung-tính của `hier_causal_policy` cũng đã ba-seed (job 5511229): huấn luyện BẬT so với TẮT dịch AP inductive tối đa $|\Delta| = 1.5\mathrm{e}{-3}$ (trung bình ≈ −1.7e-4, lẫn-dấu) $\ll$ ±3.5e-3 độ lệch chuẩn seed — AP-trung-tính trong phạm vi nhiễu seed, được bảo đảm theo cấu trúc bởi §3.4 (chính sách không bao giờ đưa vào loss AP) nhưng *không* giống-hệt-byte, vì bật op trong lúc huấn luyện làm dịch luồng RNG. Mục duy nhất còn chỉ-seed-42 là kiểm tra sanity hình-dạng-vòng-đời liên-dataset của §6.5.

---

## 9. Kết luận

SR-GNN tái khung (reframe) một quyết định thiết kế thường được coi là hiển nhiên — huấn luyện biểu diễn trên loss của tác vụ — và cho thấy điều ngược lại là tốt hơn cho dự đoán liên kết thời gian theo inductive. Giữ backbone stop-gradient khỏi link head và định hình nó bằng tiết kiệm mang lại **+13.5 điểm** AP inductive so với baseline khớp-giao-thức tốt nhất trên CoEdit và cân bằng tốt nhất trên Wikipedia và MOOC — TGAT nhỉnh hơn nó theo inductive trên Wikipedia (0.998 so với 0.996) nhưng sụp đổ theo transductive ở đó (0.658), trong khi SR-GNN giữ vững cả hai giao thức — tất cả ở ba seed dưới một giao thức đã-được-kiểm-toán-rò-rỉ. Cùng sự tách rời đó mua một SCM vòng đời năm-trạng-thái trung thực, có thể can thiệp, không tốn gì cho bộ dự đoán (được chứng minh bằng tính bất biến điểm số), hỗ trợ các phản thực đơn điệu đảo ngược được, và phát ra một tín hiệu nhất quán nội tại ổn định mà phạm vi của nó chúng tôi báo cáo một cách trung thực. Chúng tôi xem decoupling-by-construction (tách-rời-theo-cấu-trúc) là một nguyên lý có thể tái sử dụng cho các mô hình đồ thị thời gian phải tổng quát hóa sang các thực thể chưa thấy trong khi vẫn có thể được thẩm vấn.

---

## References

Alemi, A. A., Fischer, I., Dillon, J. V., & Murphy, K. (2017). Deep Variational Information Bottleneck. *International Conference on Learning Representations (ICLR)*. arXiv:1612.00410.

Chen, X., & He, K. (2021). Exploring Simple Siamese Representation Learning. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 15750–15758.

Cong, W., Zhang, S., Kang, J., Yuan, B., Wu, H., Zhou, X., Tong, H., & Mahdavi, M. (2023). Do We Really Need Complicated Model Architectures for Temporal Networks? *International Conference on Learning Representations (ICLR)*. [GraphMixer]

Hawkes, A. G. (1971). Spectra of Some Self-Exciting and Mutually Exciting Point Processes. *Biometrika*, 58(1), 83–90.

Huang, S., Poursafaei, F., Danovitch, J., Fey, M., Hu, W., Rossi, E., Leskovec, J., Bronstein, M., Rabusseau, G., & Rabbany, R. (2023). Temporal Graph Benchmark for Machine Learning on Temporal Graphs. *Advances in Neural Information Processing Systems (NeurIPS)*. [TGB; fair-negative protocol]

Jacovi, A., & Goldberg, Y. (2020). Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness? *Annual Meeting of the Association for Computational Linguistics (ACL)*, pp. 4198–4205.

Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *International Conference on Learning Representations (ICLR)*. [VAE]

Kumar, S., Zhang, X., & Leskovec, J. (2019). Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, pp. 1269–1278. [JODIE]

Mei, H., & Eisner, J. (2017). The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process. *Advances in Neural Information Processing Systems (NeurIPS) 30*, pp. 6754–6764.

Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.

Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M. (2020). Temporal Graph Networks for Deep Learning on Dynamic Graphs. *ICML Workshop on Graph Representation Learning and Beyond (GRL+)*. [TGN]

Rudin, C. (2019). Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead. *Nature Machine Intelligence*, 1(5), 206–215.

Tishby, N., Pereira, F. C., & Bialek, W. (2000). The Information Bottleneck Method. *arXiv:physics/0004057*. [orig. Proc. 37th Allerton Conf. on Communication, Control and Computing, 1999]

Trivedi, R., Farajtabar, M., Biswal, P., & Zha, H. (2019). DyRep: Learning Representations over Dynamic Graphs. *International Conference on Learning Representations (ICLR)*. [DyRep]

Wang, Y., Chang, Y.-Y., Liu, Y., Leskovec, J., & Li, P. (2021). Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks. *International Conference on Learning Representations (ICLR)*. [CAWN]

Welford, B. P. (1962). Note on a Method for Calculating Corrected Sums of Squares and Products. *Technometrics*, 4(3), 419–420.

Xu, D., Ruan, C., Korpeoglu, E., Kumar, S., & Achan, K. (2020). Inductive Representation Learning on Temporal Graphs. *International Conference on Learning Representations (ICLR)*. [TGAT]

Ying, R., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019). GNNExplainer: Generating Explanations for Graph Neural Networks. *Advances in Neural Information Processing Systems (NeurIPS) 32*, pp. 9240–9251.

---

## Phụ lục A — Xuất xứ bằng chứng (Evidence provenance)

Mọi đường dẫn tương đối với `SR-GNN/experiments/results/` trừ khi ghi chú khác. Ba-seed = {42, 1, 7}.

- **SR-GNN config B, 3-seed:** `v3_3_coedit_ARM_B_publishable_3seed.json` (ind 0.9885, trans 0.9985); `v3_3_wikipedia_ARM_B_publishable_3seed.json` (ind 0.9959, trans 0.9993); `v3_3_mooc_ARM_B_publishable_3seed_rerun.json` (ind 0.9978, trans 0.9988). *Lưu ý: các trường `*_std` được lưu bên trong các JSON này là population std (÷n); mọi ± được báo cáo trong paper đều được tính lại là sample std (n−1) từ các giá trị theo từng seed, theo quy ước ở masthead.*
- **Baselines, B-protocol, 3-seed:** `baselines/baselines_coedit_Bprotocol.json`, `baselines/baselines_wikipedia_Bprotocol.json`, `baselines/baselines_mooc.json`.
- **Ablation tách rời (B so với C, ba seed {42,1,7}):** B từ `v3_3_coedit_ARM_B_publishable_3seed.json` (ind 0.9885 ± 0.0035); C từ `v3_3_coedit_ARM_C_correct_3seed.json` (ind 0.7672 ± 0.0107, theo từng seed [0.7792, 0.7639, 0.7585], trans 0.9609; job 5503786). Δ(B−C) = +22.1 ± 1.4 pp, theo từng seed [0.206, 0.225, 0.234]. (Dump seed-42 trước đó: `v3_3_3arm_coedit_B_decoupled_s42.json` 0.9871, `v3_3_3arm_coedit_C_correct_s42.json` 0.7655.)
- **causal_batch A/B (config B đầy đủ, ba seed):** BẬT từ `v3_3_coedit_ARM_B_publishable_3seed.json` (ind 0.9885 ± 0.0035, trans 0.9985); TẮT từ `v3_3_coedit_B_causalOFF_3seed.json` (ind 0.9312 ± 0.0027, trans 0.9920; job 5503786). Δ = +5.7 ± 0.2 pp ind / +0.65 pp trans. (A/B đơn-seed được lược trước đó: `v3_3_causal_ab_coedit_cbON.json` 0.7907, `v3_3_causal_ab_coedit_cbOFF.json` 0.7462, job 5467100.)
- **hier_causal_policy A/B (ba seed {1,7,42}, job 5511229):** BẬT từ `v3_3_coedit_ARM_B_publishable_3seed.json`, TẮT từ `v3_3_coedit_B_hcpOFF_3seed.json`. Δ(BẬT−TẮT) inductive theo từng seed = +1.04e-4 (s1) / +9.29e-4 (s7) / −1.53e-3 (s42); tối đa $|\Delta_{\text{ind}}| = 1.5\mathrm{e}{-3}$, trung bình ≈ −1.7e-4, lẫn-dấu; tối đa $|\Delta_{\text{trans}}| = 1.6\mathrm{e}{-4}$; so với ±3.5e-3 độ lệch chuẩn seed (ind). AP-trung-tính trong phạm vi nhiễu seed (không giống-hệt-byte: jitter RNG thời điểm huấn luyện). (Dump seed-42 trước đó: `v3_3_hcp_coedit_ON_s42.json` 0.9871 so với `_OFF_s42.json` 0.9872, job 5471271.)
- **Dàn phản thực (Counterfactual battery, ba seed {42,1,7}, config B / cbON):** `experiments/LAB/v3_3/fsm_intervene.py` trên `faithfulness_coedit_v3_hier_hv2_let0.5_s{42,1,7}_cbON.npz` (N=12000 mỗi seed; s1/s7 huấn luyện mới dạng config-B cbON, job 5506704). Thang tồn tại chính xác mọi seed; do(DEATH)→P(edge)↓ ≥99% mọi seed; do(noop)/đảo-ngược Δ=0 chính xác mọi seed; quỹ đạo DECAY→do(slope+)→REINFORCE = 0.9999 ± 0.0002 (theo seed 1.0/1.0/0.9996); các dấu đáp-ứng-theo-liều ổn định ba seed (rate→REINFORCE +0.028±0.004, rate→DEATH −0.068±0.007, slope→REINFORCE +0.489±0.003, true_occ→BIRTH −0.0065). Phân rã giết REINFORCE→DEATH (ba seed, tái diễn true_occ≥2; nguồn `cf_kill_REINFORCE_3seed.json`): do(rate=dead)→DEATH cô lập 0.597±0.028 (theo seed 0.574/0.588/0.629), do(staleness=high)→DEATH cô lập 0.482±0.007 (0.477/0.479/0.489), tất-cả-tác-nhân-chết do(rate=dead,slope=falling,staleness=high)→DEATH 0.999±0.002 (1.0/1.0/0.997). Thay dump seed-42 không tái lập được (83–86% / 100%). Bus [ml→PM] 2026-06-03 / 2026-06-06.
- **Hình dạng vòng đời liên-dataset (sanity faithfulness seed 42, job 5506705):** `faithfulness_coedit_v3_hier_hv2_let0.5_s42_cbON.npz`, `faithfulness_wikipedia_v3_hier_hv2_cb_let0.5_s42.npz`, `faithfulness_mooc_v3_hier_hv2_cb_let0.5_s42.npz`. Argmax `s_t1_cal` trên tập con tái diễn (true_occ≥2): CoEdit REINFORCE .35/DECAY .62/DEATH .02; Wikipedia .42/.43/.16 (entropy 1.31); MOOC .62/.08/.30 (nặng-BIRTH). Tất cả không-suy-biến, phù-hợp-dataset (§6.5).
- **Độ tin cậy (WC-CONF grounded-init, 3-seed):** `experiments/LAB/v3_3/results/wc_grnd/wc_conf_calib_grnd_coedit_s{42,1,7}_summary.json` — self-consistency AUC 0.9985±0.0015, external posMiss10 AUC 0.405±0.484 (sample std, n−1). Jobs 5503466/5503467.
- **Integrity audit:** bus [ml→PM] 2026-06-06. **Anti-leak re-gate:** job 5450095. **Architecture of record:** `v3_3_ARCHITECTURE_CURRENT.md` (đã kiểm chứng 2026-06-06).

*Mục tiêu word count ≤ 8000. Các mục cần thêm bằng chứng trước khi nộp được liệt kê trong báo cáo PAPER→PM.*
