from core import run_llm

if __name__ == "__main__":
    while True:
        user_input = input("Ask me anything about being a Software Engineer!\n")

        if not user_input:
            print("Your query is empty. Please ask again.")
            continue
        else:
            break

    print(run_llm(user_input))
