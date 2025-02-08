import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
	data = pd.read_csv('mini-data.csv', header=None)
	data = data.drop(columns=0)
	data[1] = data[1].map({'M': 1, 'B': 0})

	X = data.drop(columns=1)
	y = data[1]
	X_train, X_test, y_train, y_test = train_test_split(
		X, y,
		test_size=0.2,
		stratify=y,
		random_state=42,
		shuffle=True
	)

	# Normalize features using standardization
	scaler = StandardScaler()
	X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
	X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

	# Combine labels with normalized features
	train_dataset = pd.concat([y_train.reset_index(drop=True), X_train], axis=1)
	test_dataset = pd.concat([y_test.reset_index(drop=True), X_test], axis=1)

	# Save to CSV files
	train_dataset.to_csv('training-dataset.csv', index=False)
	test_dataset.to_csv('test-dataset.csv', index=False)

	print("Preprocessing complete!")
	print(f"Training samples: {len(train_dataset)}")
	print(f"Test samples: {len(test_dataset)}")


if __name__ == "__main__":
	main()