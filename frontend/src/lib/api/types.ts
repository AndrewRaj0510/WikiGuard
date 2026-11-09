/**
 * Names for the API's shapes. They come from `schema.d.ts`, which is generated from the API's
 * OpenAPI schema (`npm run types`), so a change in the API shows up here as a type error.
 */
import type { components } from "./schema";

type Schemas = components["schemas"];

export type SessionInfo = Schemas["SessionInfo"];
export type Role = SessionInfo["role"];
export type GateStatus = Schemas["GateStatus"];
export type HeartbeatReply = Schemas["HeartbeatReply"];
export type EditSummary = Schemas["EditSummary"];
export type EditPage = Schemas["EditPage"];
export type EditDetail = Schemas["EditDetail"];
export type ScoreInfo = Schemas["ScoreInfo"];
export type ExplainReply = Schemas["ExplainReply"];
export type Decision = NonNullable<EditSummary["decision"]>;
export type ReviewReply = Schemas["ReviewReply"];
export type ModelEntry = Schemas["ModelEntry"];
export type ModelList = Schemas["ModelList"];
export type HealthSummary = Schemas["HealthSummary"];
export type BreakerInfo = Schemas["BreakerInfo"];

/**
 * The admin routes are left out of the public OpenAPI schema on purpose (PLAN section 8), so
 * these two are written by hand from services/api/wikiguard_api/models.py. Keep them in step.
 */
export interface AdminRequestInfo {
  id: number;
  kind: "retrain" | "promote" | "rollback" | string;
  status: string;
  reason: string;
  requested_by: string;
  requested_at: string;
  picked_up_at: string | null;
  finished_at: string | null;
  details: Record<string, unknown>;
}

export interface AdminRequestList {
  items: AdminRequestInfo[];
}

/** The review queue's server-side filter (DEC-053). */
export type EditorFilter = "all" | "logged_out" | "registered";
