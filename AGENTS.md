# AI Coding Guidelines for This Project (Python)

## General Principles

* Follow all rules strictly.
* Prefer clarity and readability over cleverness.
* Code must be consistent across the entire project.

---

## Formatting Rules

* NEVER exceed 80 characters per line.
* ALWAYS follow PEP 8 strictly.
* Use consistent indentation (4 spaces).
* Avoid overly long functions (prefer small, focused ones).

---

## File-Level Requirements

* At the very top of every file, include a comment in English
  explaining what the file does.

Example:

# This module handles user authentication and session management.

---

## Docstrings (MANDATORY)

Every function, class, and module MUST have a docstring in the
following format:

EN: <English description>

PL: <Polish description>

Rules:

* Always include both languages.
* Be precise and descriptive.
* Describe inputs, outputs, and behavior.

Example:

def add(a: int, b: int) -> int:
"""
EN:
Adds two integers and returns the result.

```
PL:
Dodaje dwie liczby całkowite i zwraca wynik.
"""
return a + b
```

---

## Comments (VERY IMPORTANT)

* Add MANY one-line comments in Polish explaining the code step-by-step.
* Explain intent, not just obvious operations.
* Comments should appear above or next to important lines.

Example:

# Pobieramy dane użytkownika z bazy

user = get_user(user_id)

# Sprawdzamy czy użytkownik istnieje

if not user:
raise ValueError("User not found")

---

## Code Style

* Use clear and descriptive variable names.
* Avoid single-letter variables (except simple loops).
* Do not use `any` or untyped values if avoidable.
* Prefer explicit over implicit logic.

---

## Function Design

* Each function should do ONE thing.
* Keep functions short and readable.
* Use type hints everywhere.

---

## What to Avoid

* No overly complex one-liners.
* No hidden side effects.
* No mixing responsibilities in one function.
* No undocumented code.

---

## Priority Order (IMPORTANT)

If rules conflict, follow this order:

1. Clarity and readability
2. PEP 8 compliance
3. Line length limit (80 chars)
4. Consistency with existing code

---

## Behavior Expectations for AI

* Always follow these rules when generating or modifying code.
* If unsure, choose the more readable and explicit solution.
* Do not skip comments or docstrings under any circumstances.

---
