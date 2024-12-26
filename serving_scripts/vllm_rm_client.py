# curl -X POST -H "Content-Type: application/json" http://server1.zhaojunhua.org:89/v1/api/aflow/evaluation -d '{"dataset_type": "mbpp","queries": ["Given a square matrix of size N*N given as a list of lists, where each cell is associated with a specific cost. A path is defined as a specific sequence of cells that starts from the top-left cell move only right or down and ends on bottom right cell. We want to find a path with the maximum average over all existing paths. Average is computed as total cost divided by the number of cells visited in the path.\n\ndef maxAverageOfPath(cost):"], "codes":["ANALYZE_REQUIREMENTS_PROMPT = \"\"\"\nAnalyze the given problem and identify the key requirements, constraints, and expected behavior of the solution. Provide a concise summary of these aspects to guide the code generation process.\n\"\"\"\n\nGENERATE_CODE_PROMPT = \"\"\"\nGenerate a Python function to solve the given problem. Ensure the function name matches the entry point specified. Include necessary imports and helper functions. Provide a clear and efficient solution. Focus on correctness and optimal performance. Consider the provided requirements in your implementation.\n\"\"\"\n\nVALIDATE_AND_REFINE_PROMPT = \"\"\"\nReview the given solution for the problem. Validate if it meets all the requirements and constraints. If any issues are found, refine the solution to address them. Ensure the refined solution is complete, efficient, and adheres to best coding practices.\n\"\"\"\n\nIMPROVE_CODE_PROMPT = \"\"\"\nThe previous solution failed to pass the tests. Please analyze the error and provide an improved version of the code. Focus on fixing the specific issues mentioned in the error message while maintaining the overall structure and logic of the function. Ensure that your solution is complete and addresses all aspects of the problem, including the provided requirements.\n\"\"\"\nasync def __call__(self, problem: str, entry_point: str):\n        requirements = await self.custom(input=problem, instruction=prompt_custom.ANALYZE_REQUIREMENTS_PROMPT)\n        \n        solutions = []\n        for _ in range(3):\n            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.GENERATE_CODE_PROMPT + f\"\\nRequirements: {requirements['response']}\")\n            solutions.append(solution['response'])\n        \n        # New step: Validate and refine solutions\n        refined_solutions = []\n        for solution in solutions:\n            refined_solution = await self.custom(input=f\"Problem: {problem}\\nRequirements: {requirements['response']}\\nSolution: {solution}\", instruction=prompt_custom.VALIDATE_AND_REFINE_PROMPT)\n            refined_solutions.append(refined_solution['response'])\n        \n        best_solution = await self.sc_ensemble(solutions=refined_solutions, problem=problem)\n        \n        test_result = await self.test(problem=problem, solution=best_solution['response'], entry_point=entry_point)\n        \n        if test_result['result']:\n            return test_result['solution'], self.llm.cost_manager.total_cost\n        else:\n            improved_solution = await self.custom(input=f\"Problem: {problem}\\nRequirements: {requirements['response']}\\nFailed solution: {best_solution['response']}\\nError: {test_result['solution']}\", instruction=prompt_custom.IMPROVE_CODE_PROMPT)\n            return improved_solution['response'], self.llm.cost_manager.total_cost\n\n"]}'