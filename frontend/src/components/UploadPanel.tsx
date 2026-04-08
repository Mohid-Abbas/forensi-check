import type { CSSProperties } from "react";

type Props = {
  onSelect: (file: File) => void;
  isLoading: boolean;
};

export function UploadPanel({ onSelect, isLoading }: Props) {
  return (
    <div style={styles.box}>
      <label htmlFor="file-input" style={styles.label}>
        Drag and drop an image or click to select
      </label>
      <input
        id="file-input"
        type="file"
        accept="image/*"
        disabled={isLoading}
        onChange={(event) => {
          const next = event.target.files?.[0];
          if (next) onSelect(next);
        }}
      />
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  box: {
    border: "1px dashed #7a7a7a",
    borderRadius: 8,
    padding: 20,
    marginBottom: 16
  },
  label: {
    display: "block",
    marginBottom: 8
  }
};
