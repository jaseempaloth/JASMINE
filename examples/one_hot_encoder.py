import numpy as np
from jasmine.preprocessing import OneHotEncoder

def main():
    data = np.array([
        [1, 'Male', 'India'],
        [2, 'Female', 'USA'],
        [3, 'Female', 'Canada'],
        [4, 'Male', 'China']
    ])

    print("Original Data:")
    print(data)

    encoder = OneHotEncoder()
    transformed_data = encoder.fit_transform(data)

    print("Transformed Data")
    print(transformed_data)

if __name__ == "__main__":
    main()
