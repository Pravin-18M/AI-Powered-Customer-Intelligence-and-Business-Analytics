import urllib.request

# Replace this with the raw GitHub URL to your committed .joblib file.
JOBLIB_URL = ''
LOCAL_MODEL_FILE = Path('customer_intelligence_models.joblib')

if JOBLIB_URL.strip():
    downloaded_file = Path('downloaded_customer_intelligence_models.joblib')
    urllib.request.urlretrieve(JOBLIB_URL, downloaded_file)
    reusable_bundle = joblib.load(downloaded_file)
else:
    reusable_bundle = joblib.load(LOCAL_MODEL_FILE)

print('Loaded model format:', reusable_bundle['format_version'])
print('Available artifacts:', sorted(reusable_bundle.keys()))

# Example: reuse the revenue model with one already-engineered customer record.
example_customer = pd.DataFrame([{
    'total_orders': 8,
    'average_order_value': 2500.0,
    'product_variety': 5,
    'category_variety': 3,
    'recency_days': 30,
    'average_rating': 4.2,
    'average_shipping_days': 3.0,
    'return_rate': 0.10,
}])
if 'revenue_model' in reusable_bundle:
    revenue_features = reusable_bundle['clv_feature_columns']
    predicted_revenue = reusable_bundle['revenue_model'].predict(example_customer[revenue_features])[0]
    print(f'Predicted historical-revenue proxy: {predicted_revenue:,.2f}')
else:
    print('The downloaded bundle does not contain a revenue model.')

# Example: assign a segment using only saved preprocessing and model artifacts.
segment_features = reusable_bundle['segment_feature_columns']
segment_input = example_customer.reindex(columns=segment_features, fill_value=0)
segment_lower = pd.Series(reusable_bundle['segment_lower_bounds'])
segment_upper = pd.Series(reusable_bundle['segment_upper_bounds'])
segment_input = segment_input.clip(lower=segment_lower, upper=segment_upper, axis=1)
segment_number = reusable_bundle['segment_model'].predict(reusable_bundle['segmentation_scaler'].transform(segment_input))[0]
print(f'Assigned customer segment: {segment_number}')
