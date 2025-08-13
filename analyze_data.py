import pandas as pd
import numpy as np

df = pd.read_csv('dataset/seller_Order_Region_processed.csv')
df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])

# Kiểm tra các ngày đầu tiên theo region
print('=== KIỂM TRA DỮ LIỆU THEO THỜI GIAN VÀ REGION ===')
print('Ngày đầu tiên:', df['order date (DateOrders)'].min())
print('Ngày cuối cùng:', df['order date (DateOrders)'].max())
print('Tổng số ngày khác biệt:', (df['order date (DateOrders)'].max() - df['order date (DateOrders)'].min()).days + 1)
print('Số dòng dữ liệu thực tế:', len(df))
print()

# Kiểm tra các region
print('Số lượng region unique:', df['Order Region_encoded'].nunique())
print('Các region:', sorted(df['Order Region_encoded'].unique()))
print()

# Kiểm tra dữ liệu theo ngày đầu tiên
print('=== DỮ LIỆU 10 NGÀY ĐẦU TIÊN ===')
first_10_days = df[df['order date (DateOrders)'] <= '2015-01-10'].groupby('order date (DateOrders)')['Order Region_encoded'].apply(list).head(10)
for date, regions in first_10_days.items():
    print(f'{date.date()}: Có {len(regions)} dòng từ regions {set(regions)}')
print()

# Kiểm tra tỷ lệ missing data
date_range = pd.date_range(start=df['order date (DateOrders)'].min(), end=df['order date (DateOrders)'].max(), freq='D')
total_expected = len(date_range) * df['Order Region_encoded'].nunique()
print('=== PHÂN TÍCH MISSING DATA ===')
print(f'Số ngày expected: {len(date_range)}')
print(f'Số region: {df["Order Region_encoded"].nunique()}')
print(f'Tổng dòng expected (ngày × region): {total_expected}')
print(f'Tổng dòng thực tế: {len(df)}')
print(f'Tỷ lệ missing: {(total_expected - len(df)) / total_expected * 100:.1f}%')

# Kiểm tra region nào có nhiều dữ liệu nhất
print('\n=== PHÂN TÍCH THEO REGION ===')
region_counts = df.groupby('Order Region_encoded').size().sort_values(ascending=False)
print('Top 5 regions có nhiều dữ liệu nhất:')
for region, count in region_counts.head().items():
    print(f'Region {region}: {count} dòng ({count/len(df)*100:.1f}%)')
print()

print('Top 5 regions có ít dữ liệu nhất:')
for region, count in region_counts.tail().items():
    print(f'Region {region}: {count} dòng ({count/len(df)*100:.1f}%)')
