graph TD
    A[Start] --> B[Collect images]
    B --> C{Clustered?}
    C -->|No| D[Cluster images]
    C -->|Yes| E[Generate images]
    D --> E
    E --> F[Select images]
    F --> G[Build 3D model]
    G --> H[Evaluate 3D model]
    H --> I{Satisfactory?}
    I -->|No| J[Adjust parameters]
    J --> B
    I -->|Yes| K[End]
