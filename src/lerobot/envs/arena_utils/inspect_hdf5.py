import h5py
import numpy as np


def print_hdf5_structure(file_path, max_items=10):
    """
    Load and display the structure and main content of an HDF5 file.

    Args:
        file_path: Path to the HDF5 file
        max_items: Maximum number of items to display for arrays
    """
    print(f"Loading HDF5 file: {file_path}\n")

    with h5py.File(file_path, "r") as f:
        print("=" * 80)
        print("HDF5 File Structure")
        print("=" * 80)

        def print_structure(name, obj, indent=0):
            """Recursively print the structure of the HDF5 file."""
            prefix = "  " * indent
            if isinstance(obj, h5py.Group):
                print(f"{prefix}📁 Group: {name}/")
                # Print attributes
                if obj.attrs:
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"{prefix}  └─ @{attr_name}: {attr_value}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{prefix}📄 Dataset: {name}")
                print(f"{prefix}  ├─ Shape: {obj.shape}")
                print(f"{prefix}  ├─ Dtype: {obj.dtype}")
                print(f"{prefix}  └─ Size: {obj.size} elements")

                # Print attributes
                if obj.attrs:
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"{prefix}    └─ @{attr_name}: {attr_value}")

        # Print root level info
        print(f"\nRoot group keys: {list(f.keys())}\n")

        # Root attributes
        if f.attrs:
            print("Root Attributes:")
            for attr_name, attr_value in f.attrs.items():
                print(f"  @{attr_name}: {attr_value}")
            print()

        # Recursively visit all items
        f.visititems(print_structure)

        print("\n" + "=" * 80)
        print("Sample Data Content")
        print("=" * 80)

        # Display sample data from datasets
        def print_data_samples(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"\n📊 {name}:")
                print(f"   Shape: {obj.shape}, Dtype: {obj.dtype}")

                data = obj[:]

                # Show sample based on dimensionality
                if len(obj.shape) == 0:  # Scalar
                    print(f"   Value: {data}")
                elif len(obj.shape) == 1:  # 1D array
                    if obj.shape[0] <= max_items:
                        print(f"   Data: {data}")
                    else:
                        print(f"   First {max_items} items: {data[:max_items]}")
                        print(f"   Last {max_items} items: {data[-max_items:]}")
                        print(
                            f"   Min: {np.min(data)}, Max: {np.max(data)}, Mean: {np.mean(data)}"
                        )
                else:  # Multi-dimensional
                    print(
                        f"   First item shape: {data[0].shape if len(data) > 0 else 'N/A'}"
                    )
                    if len(data) > 0:
                        print(f"   First item sample:\n{data[0]}")
                        if len(data) > 1:
                            print(f"   Last item sample:\n{data[-1]}")

                    # Stats for numeric data
                    if np.issubdtype(obj.dtype, np.number):
                        print(
                            f"   Stats - Min: {np.min(data)}, Max: {np.max(data)}, Mean: {np.mean(data):.4f}"
                        )

        f.visititems(print_data_samples)

        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)

        # Count groups and datasets
        num_groups = 0
        num_datasets = 0

        def count_items(name, obj):
            nonlocal num_groups, num_datasets
            if isinstance(obj, h5py.Group):
                num_groups += 1
            elif isinstance(obj, h5py.Dataset):
                num_datasets += 1

        f.visititems(count_items)

        print(f"Total Groups: {num_groups}")
        print(f"Total Datasets: {num_datasets}")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Inspect the structure and content of an HDF5 file.')
    parser.add_argument('--file-path', type=str, help='Path to the HDF5 file to inspect')
    args = parser.parse_args()
    file_path = args.file_path
    print_hdf5_structure(file_path)
