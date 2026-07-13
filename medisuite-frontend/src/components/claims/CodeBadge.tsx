import { Badge } from "@/components/ui/badge";
import type { VariantProps } from "class-variance-authority";
import { badgeVariants } from "@/components/ui/badge";

export function CodeBadge({
  code,
  confidence,
}: {
  code: string;
  confidence: number;
}) {
  const variant: NonNullable<VariantProps<typeof badgeVariants>["variant"]> =
    confidence >= 0.9 ? "default" : confidence >= 0.8 ? "secondary" : "destructive";

  return (
    <Badge variant={variant} className="font-mono">
      {code} ({Math.round(confidence * 100)}%)
    </Badge>
  );
}
