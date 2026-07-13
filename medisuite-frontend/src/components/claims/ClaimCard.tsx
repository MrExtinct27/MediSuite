import Link from "next/link";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CodeBadge } from "@/components/claims/CodeBadge";
import type { Claim } from "@/lib/types";

export function ClaimCard({ claim }: { claim: Claim }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-3">
          <span className="truncate">Claim {claim.claim_id}</span>
          <Link className="text-sm text-primary hover:underline" href={`/claims/${claim.claim_id}`}>
            View
          </Link>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="text-sm text-muted-foreground">
          {claim.patient?.name} • {claim.service_date}
        </div>
        <div className="flex flex-wrap gap-2">
          {claim.diagnosis_codes?.slice(0, 3).map((d) => (
            <CodeBadge key={d.code} code={d.code} confidence={d.confidence} />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
