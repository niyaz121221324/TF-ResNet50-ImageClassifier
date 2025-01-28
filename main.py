from model import create_new_model, fine_tune_model

def main():
    """Main menu for creating or fine-tuning a model."""
    while True:
        print("1. Create a new model")
        print("2. Fine-tune an existing model")
        print("3. Exit")
        choice = input("Select an option (1/2/3): ")

        if choice == '1':
            create_new_model()
        elif choice == '2':
            fine_tune_model()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()