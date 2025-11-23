document.addEventListener("DOMContentLoaded", () => {
    const block = document.querySelector(".frontpage-code");
    if (!block) return;
  
    const button = document.createElement("button");
    button.innerText = "Copy";
    button.className = "copy-btn";
  
    button.addEventListener("click", () => {
      const code = block.innerText;
      navigator.clipboard.writeText(code).then(() => {
        button.innerText = "Copied!";
        setTimeout(() => {
          button.innerText = "Copy";
        }, 1200);
      });
    });
  
    const container = block.parentElement;
    container.style.position = "relative";
    container.appendChild(button);
  });
  