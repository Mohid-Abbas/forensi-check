import type { CSSProperties } from "react";
import { useState } from "react";

type Props = {
  onSelect: (file: File) => void;
  isLoading: boolean;
};

export function UploadPanel({ onSelect, isLoading }: Props) {
  const [isDragOver, setIsDragOver] = useState(false);

  const pick = (file?: File) => {
    if (file) onSelect(file);
  };

  return (
    <div
      style={{ ...styles.box, ...(isDragOver ? styles.boxActive : {}) }}
      onDragOver={(event) => {
        event.preventDefault();
        if (!isLoading) setIsDragOver(true);
      }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={(event) => {
        event.preventDefault();
        setIsDragOver(false);
        if (isLoading) return;
        pick(event.dataTransfer.files?.[0]);
      }}
    >
      <label htmlFor="file-input" style={styles.label}>
        Drag and drop an image or click to select
      </label>
      <p style={styles.hint}>Supports JPG, PNG, WEBP. Large images are resized for analysis.</p>
      <input
        id="file-input"
        type="file"
        accept="image/*"
        disabled={isLoading}
        onChange={(event) => {
          pick(event.target.files?.[0]);
        }}
      />
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  box: {
    border: "1px dashed #9ca3af",
    borderRadius: 14,
    padding: 20,
    marginBottom: 16,
    background: "linear-gradient(180deg, #ffffff 0%, #f6f8fb 100%)",
    transition: "all 120ms ease-in-out"
  },
  boxActive: {
    border: "1px solid #2563eb",
    boxShadow: "0 0 0 3px rgba(37, 99, 235, 0.15)"
  },
  label: {
    display: "block",
    marginBottom: 8,
    fontWeight: 600
  },
  hint: {
    margin: "0 0 10px",
    color: "#4b5563",
    fontSize: 13
  }
};
