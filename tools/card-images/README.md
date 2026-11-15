# 作品卡片圖

首頁每個作品的卡片圖（`works/*.webp`）都是用這裡的程式算出來的，不是拍照，也不是 AI 畫的：

1. 用瀏覽器打開創客小幫手，讓每個作品用內建的範例圖（`64-kid.js` 的小朋友手繪風圖案）做出真正的 3D 幾何。
2. 把作品放在牛皮紙桌面上算圖：同一個角度、暖色燈光、接觸陰影、雷切焦邊、3D 列印層紋、背景景深。
3. 照片雷雕、照片透光片的範例照片是程式畫的虛構海景（`landscape()`），不用真人照片。

## 重做

在專案根目錄（網站更新後）：

```
pip install playwright numpy pillow scipy
python3 -m playwright install chromium
python3 tools/card-images/make_cards.py            # 全部重做
python3 tools/card-images/make_cards.py charm spin # 只做幾個
```

## 加新作品

在 `make_cards.py` 的 `POSE` 加一行：`flat=True` 表示平放的零件，否則用組裝好的樣子；`rot` 是轉幾度（弧度），讓圖案朝上比較好認；`ring=True` 會在吊孔加一個鑰匙圈。範例圖在網站原始碼的 `64-kid.js` 加一個手繪圖，作品卡的 `sample` 指向它，卡片圖和打開作品看到的範例圖就會一樣。
