from ollama import Client


def test_llm_capabilities():
    client = Client(host='http://localhost:11434')

    def ask_question(prompt):
        print(f"\n--- Question: {prompt} ---")
        response = client.chat(model='llama2',
                               messages=[{
                                   'role': 'user',
                                   'content': prompt
                               }])
        print("Response:", response['message']['content'])
        print("-" * 50)

    # Test medical knowledge
    medical_questions = [
        "What are the main differences between Type 1 and Type 2 diabetes? Include treatment approaches.",
        "Explain the pathophysiology of heart failure with preserved ejection fraction (HFpEF).",
        "What are the key differential diagnoses for chest pain in the emergency department?"
    ]

    # Test programming problem-solving
    programming_questions = [
        """Write a Python function that finds the maximum value in a list using three different approaches:
        1. Using built-in max() function
        2. Using sorting
        3. Using iteration with a loop
        Compare their time complexity.""",

        """Explain how you would optimize this code for finding duplicates in a large list:
        def find_duplicates(lst):
            duplicates = []
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    if lst[i] == lst[j] and lst[i] not in duplicates:
                        duplicates.append(lst[i])
            return duplicates""",
    ]

    # Test combination of medical and programming knowledge
    complex_questions = [
        """Design a simple patient monitoring system. Consider:
        1. What vital signs would you track?
        2. How would you implement alert thresholds?
        3. Write a Python class structure for this system.
        Provide a code example.""",

        """Given a dataset of patient blood pressure readings over time:
        1. How would you identify concerning trends?
        2. What statistical methods would be appropriate?
        3. Write a Python function to analyze this data.
        Include both medical reasoning and code."""
    ]

    print("Testing Medical Knowledge...")
    for question in medical_questions:
        ask_question(question)

    print("\nTesting Programming Knowledge...")
    for question in programming_questions:
        ask_question(question)

    print("\nTesting Combined Medical and Programming Knowledge...")
    for question in complex_questions:
        ask_question(question)


if __name__ == "__main__":
    test_llm_capabilities()
