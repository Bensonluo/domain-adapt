"""Optional Gradio shell for comparing a base and adapted backend."""

from __future__ import annotations

DISCLAIMER = (
    "Scaffold only: no model backend is configured. Outputs are not medical or legal advice."
)


def compare(prompt: str, domain: str) -> tuple[str, str]:
    length = len(prompt.strip())
    message = f"[{domain}] Backend not configured. Received {length} characters. {DISCLAIMER}"
    return message, message


def build_demo():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "install the optional demo dependencies: pip install -e '.[demo]'"
        ) from exc

    with gr.Blocks(title="AdaptStack comparison scaffold") as app:
        gr.Markdown("# AdaptStack comparison scaffold\n\n" + DISCLAIMER)
        domain = gr.Dropdown(["medical", "legal"], value="medical", label="Domain")
        prompt = gr.Textbox(label="Prompt")
        submit = gr.Button("Compare")
        base = gr.Textbox(label="Base model", interactive=False)
        adapted = gr.Textbox(label="Adapted model", interactive=False)
        submit.click(compare, inputs=[prompt, domain], outputs=[base, adapted])
    return app


if __name__ == "__main__":
    build_demo().launch(server_name="127.0.0.1", share=False)
