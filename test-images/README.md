# Test images

Drop plastic-item photos here (`.jpg`, `.png`, `.webp`, `.gif`, or `.bmp`) and run:

```bash
npm run test:eval
```

For a meaningful real-world accuracy read, include a mix of:
- Different resins (PET bottles, HDPE jugs, PP cups, PS foam)
- Different contamination levels (clean, lightly soiled, heavily soiled)
- Deformed/crushed items, off-angle shots, and varied lighting

This folder's images are gitignored — only this README is tracked, so the
test set stays local to whoever is running the eval.
