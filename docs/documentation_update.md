📘 Pygame Documentation – Improvement Proposal & Contribution Guide
✅ Overview
This document aims to improve unclear or missing parts of the Pygame documentation. It includes explanations, examples, and a contribution guide that new users can easily follow.

🟦 1. Event Handling – Clear Explanation & Examples
Many event types in Pygame are not clearly explained in the documentation. Here is a simplified version:

📌 Common Event Types
Event Type	Description
QUIT	Window close button pressed
KEYDOWN	A key was pressed
KEYUP	A key was released
TEXTINPUT	Text typed by the user
MOUSEBUTTONDOWN	Mouse button clicked
MOUSEWHEEL	Mouse scroll up/down
✅ Example: Keyboard & Quit Event
import pygame

pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Event Example")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            print(f"Key Pressed: {event.key}")

pygame.quit()
🟦 2. Font Rendering & Text Size Clarification
Users often get confused about font metrics like ascent, descent, height, and text size.

✅ Example:
import pygame

pygame.init()
font = pygame.font.Font(None, 40)

print("Ascent:", font.get_ascent())
print("Descent:", font.get_descent())
print("Height:", font.get_height())
print("Text Size:", font.size("Hello"))
Property	Meaning
Ascent	Height above baseline
Descent	Space below baseline
Height	Total height of font
font.size()	Width & height of any text in pixels
🟦 3. How to Build Documentation Locally (Sphinx)
pip install sphinx
cd docs
make html   # or `make.bat html` on Windows
Open the file:
docs/_build/html/index.html

🟦 4. How to Contribute Documentation to Pygame
Fork the repository on GitHub

Clone it to your system:it clone https://github.com/your-username/pygame.
