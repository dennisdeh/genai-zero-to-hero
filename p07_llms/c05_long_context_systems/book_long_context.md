## Long Context Systems
Sometimes we can fit the entire context in the memory, if the models context window is large enough.

This means that in some cases, we can avoid the need for external memory storage and simply keep the
entire conversation history in the model's memory. This can be particularly useful for applications that
require a high degree of context awareness, such as chatbots or virtual assistants.

Upsides:
- Reduced latency
- Simplified architecture

Downsides:
- Still the context window is limited compared to what is needed for many tasks
- Inefficient to do inference on long contexts (there might be much non-important information in the context)
