import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import type { Explainability } from "@/lib/types";

export function ExplainabilityPanel({ data }: { data: Explainability }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Explainability</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <div>
          <p className="font-medium">ICD-10 reasoning</p>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-muted-foreground">
            {data.icd10_reasoning?.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
        <Separator />
        <div>
          <p className="font-medium">CPT-4 reasoning</p>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-muted-foreground">
            {data.cpt4_reasoning?.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
        <Separator />
        <div>
          <p className="font-medium">Citations</p>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-muted-foreground">
            {data.citations?.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}
